import cv2
import h5py
import numpy as np
import os
import pickle
import random
import torch
import util

from .base_ct_dataset import BaseCTDataset
from ct.ct_pe_constants import *
from scipy.ndimage.interpolation import rotate
import torch.nn.functional as F


class CTPEDataset3d(BaseCTDataset):
    def __init__(self, args, phase, is_training_set=True):
        """
        Args:
            args: Command line arguments.
            phase: one of 'train','val','test'
            is_training_set: If true, load dataset for training. Otherwise, load for test inference.
        """
        super(CTPEDataset3d, self).__init__(args.data_dir, args.img_format, is_training_set=is_training_set)
        self.radfusion_np_path = args.radfusion_np_path
        self.window_shift = args.window_shift
        self.phase = phase
        self.resize_shape = args.resize_shape
        self.is_test_mode = not args.is_training
        self.pe_types = args.pe_types

        # Augmentation
        self.crop_shape = args.crop_shape
        self.do_hflip = self.is_training_set and args.do_hflip
        self.do_vflip = self.is_training_set and args.do_vflip
        self.do_rotate = self.is_training_set and args.do_rotate
        self.do_jitter = self.is_training_set and args.do_jitter
        self.do_center_abnormality = self.is_training_set and args.do_center_pe
        self.do_contrast = self.is_training_set and args.do_contrast
        self.do_brightness =  self.is_training_set and args.do_brightness
        self.do_affine =  self.is_training_set and args.do_affine
        self.threshold_size = args.threshold_size
        self.pixel_dict = {
            'min_val': CONTRAST_HU_MIN,
            'max_val': CONTRAST_HU_MAX,
            'avg_val': CONTRAST_HU_MEAN,
            'w_center': W_CENTER_DEFAULT,
            'w_width': W_WIDTH_DEFAULT,
            'std_val': CONTRAST_HU_STD
        }

        # Load info for the CTPE series in this dataset
        with open(args.pkl_path, 'rb') as pkl_file:
            all_ctpes = pickle.load(pkl_file)
        print(len(all_ctpes))
        self.ctpe_list = [ctpe for ctpe in all_ctpes if self._include_ctpe(ctpe)]
        print(len(self.ctpe_list))
        self.positive_idxs = [i for i in range(len(self.ctpe_list)) if self.ctpe_list[i].is_positive]
        self.min_pe_slices = args.min_abnormal_slices
        self.num_slices = args.num_slices
        self.abnormal_prob = args.abnormal_prob if self.is_training_set else None
        self.use_hem = args.use_hem if self.is_training_set else None

        self.scale_by_std=args.scale_by_std

        # Map from windows to series indices, and from series indices to windows
        self.window_to_series_idx = []  # Maps window indices to series indices
        self.series_to_window_idx = []  # Maps series indices to base window index for that series
        window_start = 0
        for i, s in enumerate(self.ctpe_list):
            num_windows = len(s) // self.num_slices + (1 if len(s) % self.num_slices > 0 else 0)
            self.window_to_series_idx += num_windows * [i]
            self.series_to_window_idx.append(window_start)
            window_start += num_windows

        if args.toy:
            self.window_to_series_idx = np.random.choice(self.window_to_series_idx, args.toy_size, replace=False)

        if self.use_hem:
            # Initialize a HardExampleMiner with IDs formatted like (series_idx, start_idx)
            example_ids = []
            for window_idx in range(len(self)):
                series_idx = self.window_to_series_idx[window_idx]
                series = self.ctpe_list[series_idx]
                if not series.is_positive:
                    # Only include negative examples in the HardExampleMiner
                    start_idx = (window_idx - self.series_to_window_idx[series_idx]) * self.num_slices
                    example_ids.append((series_idx, start_idx))
            self.hard_example_miner = util.HardExampleMiner(example_ids)

    def _include_ctpe(self, pe):
        """Predicate for whether to include a series in this dataset."""
        if pe.phase != self.phase and self.phase != 'all':
            return False
        
        if pe.is_positive and pe.type not in self.pe_types:
            return False

        return True
    def __len__(self):
        return len(self.window_to_series_idx)

    def __getitem__(self, idx):
        # Choose ctpe and window within ctpe
        ctpe_idx = self.window_to_series_idx[idx]
        ctpe = self.ctpe_list[ctpe_idx]
        #print("CTPE idxs",ctpe.pe_idxs)

        if self.abnormal_prob is not None and random.random() < self.abnormal_prob:
            # Force aneurysm window with probability `abnormal_prob`.
            if not ctpe.is_positive:
                ctpe_idx = random.choice(self.positive_idxs)
                ctpe = self.ctpe_list[ctpe_idx]
            start_idx = self._get_abnormal_start_idx(ctpe, do_center=self.do_center_abnormality)
        elif self.use_hem:
            # Draw from distribution that weights hard negatives more heavily than easy examples
            ctpe_idx, start_idx = self.hard_example_miner.sample()
            ctpe = self.ctpe_list[ctpe_idx]
        else:
            # Get sequential windows through the whole series
            # TODO
            start_idx = (idx - self.series_to_window_idx[ctpe_idx]) * self.num_slices
            if self.window_shift:
                start_idx += -self.num_slices // 2
                start_idx = min(max(start_idx, 0), len(ctpe) - self.num_slices)
        if self.do_jitter:
            # Randomly jitter start offset by num_slices / 2
            start_idx += random.randint(-self.num_slices // 2, self.num_slices // 2)
            start_idx = min(max(start_idx, 0), len(ctpe) - self.num_slices)
     #   print("[get item]",idx, ctpe_idx,ctpe.study_num, start_idx)
        volume = self._load_volume(ctpe, start_idx)
        volume = self._transform2(volume)
     #   print("Volume", volume.size())
        is_abnormal,num_abnormal = self._is_abnormal(ctpe, start_idx)
        is_abnormal = torch.tensor([is_abnormal], dtype=torch.float32)
        num_abnormal = torch.tensor([num_abnormal*1.0/self.num_slices], dtype=torch.float32)
        #print(volume.size())
        # Pass series info to combine window-level predictions
        target = {'is_abnormal': is_abnormal,
                  'study_num': ctpe.study_num,
                  'dset_path': str(ctpe.study_num),
                  'slice_idx': start_idx,
                  'series_idx': ctpe_idx,
                  'num_abnormal':num_abnormal}
      #  print(target['is_abnormal'], target['study_num'])

        return volume, target

    def get_series_label(self, series_idx):
        """Get a floating point label for a series at given index."""
        return float(self.ctpe_list[series_idx].is_positive)

    def get_series(self, study_num):
        """Get a series with specified study number."""
        for ctpe in self.ctpe_list:
            if ctpe.study_num == study_num:
                return ctpe
        return None



    def update_hard_example_miner(self, example_ids, losses):
        """Update HardExampleMiner with set of example_ids and corresponding losses.

        This should be called at the end of every epoch.

        Args:
            example_ids: List of example IDs which were used during training.
            losses: List of losses for each example ID (must be parallel to example_ids).
        """
        example_ids = [(series_idx, start_idx) for series_idx, start_idx in example_ids
                       if series_idx not in self.positive_idxs]
        if self.use_hem:
            self.hard_example_miner.update_distribution(example_ids, losses)

    def _get_abnormal_start_idx(self, ctpe, do_center=True):
        """Get an abnormal start index for num_slices from a series.

        Args:
            ctpe: CTPE series to sample from.
            do_center: If true, center the window on the abnormality.

        Returns:
            Randomly sampled start index into series.
        """
        abnormal_bounds = (min(ctpe.pe_idxs), max(ctpe.pe_idxs))

        # Get actual slice number
        if do_center:
            # Take a window from center of abnormal region
            center_idx = sum(abnormal_bounds) // 2
            start_idx = max(0, center_idx - self.num_slices // 2)
        else:
            # Randomly sample num_slices from the abnormality (taking at least min_pe_slices).
            start_idx = random.randint(abnormal_bounds[0] - self.num_slices + self.min_pe_slices,
                                       abnormal_bounds[1] - self.min_pe_slices + 1)

        return start_idx

    def _load_volume(self, ctpe, start_idx):
        """Load num_slices slices from a CTPE series, starting at start_idx.

        Args:
            ctpe: The CTPE series to load slices from.
            start_idx: Index of first slice to load.

        Returns:
            volume: 3D NumPy arrays for the series volume.
        """
        if self.img_format == 'png':
            raise NotImplementedError('No support for PNGs in our HDF5 files.')
        # Modified
        with h5py.File(os.path.join(self.data_dir, 'data.hdf5'), 'r') as hdf5_fh:
            #print("HDF5", ctpe.study_num, start_idx, self.num_slices)
            #print(hdf5_fh.keys())
            volume = hdf5_fh[self.radfusion_np_path+str(ctpe.study_num)][start_idx:start_idx + self.num_slices]
            # Pick a pre-amble and post amble number of samples, no other change!
            #start_idx1 = start_idx -self.num_slices // 2
            #end_idx1 = start_idx1+2*self.num_slices
            #volume = hdf5_fh["C:\\Users\\preet\\Documents\\multimodalpulmonaryembolismdataset\\all\\"+str(ctpe.study_num)][start_idx1:end_idx1]
        return volume

    def _is_abnormal(self, ctpe, start_idx):
        """Check whether a window from `ctpe` starting at start_idx includes an abnormality.

        Args:
            ctpe: CTPE object to check for any abnormality.

        Returns:
            True iff (1) ctpe contains an aneurysm and (2) abnormality is big enough.
        """
        if ctpe.is_positive:
            abnormal_slices = [i for i in ctpe.pe_idxs if start_idx <= i < start_idx + self.num_slices]
            is_abnormal = len(abnormal_slices) >= self.min_pe_slices
            num_abnormal = len(abnormal_slices)
        else:
            is_abnormal = False
            num_abnormal = 0

        return is_abnormal, num_abnormal

    def _crop(self, volume, x1, y1, x2, y2):
        """Crop a 3D volume (before channel dimension has been added)."""
        volume = volume[:, y1: y2, x1: x2]

        return volume

    def _rescale(self, volume, interpolation=cv2.INTER_AREA):
        return util.resize_slice_wise(volume, tuple(self.resize_shape), interpolation)
# Modified
    def _pad(self, volume):
        """Pad a volume to make sure it has the expected number of slices.
        Pad the volume with slices of air.

        Args:
            volume: 3D NumPy array, where slices are along depth dimension (un-normalized raw HU).

        Returns:
            volume: 3D NumPy array padded/cropped to have the expected number of slices.
        """

        def add_padding(volume_, pad_value=AIR_HU_VAL):
            """Pad 3D volume with air on both ends to desired number of slices.
            Args:
                volume_: 3D NumPy ndarray, where slices are along depth dimension (un-normalized raw HU).
                pad_value: Constant value to use for padding.
            Returns:
                Padded volume with depth args.num_slices. Extra padding voxels have pad_value.
            """
            num_pad = self.num_slices - volume_.shape[0]
            volume_ = np.pad(volume_, ((0, num_pad), (0, 0), (0, 0)), mode='constant', constant_values=pad_value)

            return volume_

        volume_num_slices = volume.shape[0]

        if volume_num_slices < self.num_slices:
            volume = add_padding(volume, pad_value=AIR_HU_VAL)
        elif volume_num_slices > self.num_slices:
            # Choose center slices
            start_slice = (volume_num_slices - self.num_slices) // 2
            volume = volume[start_slice:start_slice + self.num_slices, :, :]

        return volume
    # Modified
    def _pad2(self, volume):
        """Pad a volume to make sure it has the expected number of slices.
        Pad the volume with slices of air.

        Args:
            volume: 3D NumPy array, where slices are along depth dimension (un-normalized raw HU).

        Returns:
            volume: 3D NumPy array padded/cropped to have the expected number of slices.
        """

        def add_padding(volume_, pad_value=AIR_HU_VAL):
            """Pad 3D volume with air on both ends to desired number of slices.
            Args:
                volume_: 3D NumPy ndarray, where slices are along depth dimension (un-normalized raw HU).
                pad_value: Constant value to use for padding.
            Returns:
                Padded volume with depth args.num_slices. Extra padding voxels have pad_value.
            """
            num_pad = 2*self.num_slices - volume_.shape[0]
            volume_ = np.pad(volume_, ((0, num_pad), (0, 0), (0, 0)), mode='constant', constant_values=pad_value)

            return volume_

        volume_num_slices = volume.shape[0]

        if volume_num_slices < 2*self.num_slices:
            volume = add_padding(volume, pad_value=AIR_HU_VAL)
        elif volume_num_slices > 2*self.num_slices:
            # Choose center slices
            start_slice = (volume_num_slices - self.num_slices) // 2
            volume = volume[start_slice:start_slice + self.num_slices, :, :]

        return volume

    def _transform(self, inputs):
        """Transform slices: resize, random crop, normalize, and convert to Torch Tensor.

        Args:
            inputs: 2D/3D NumPy array (un-normalized raw HU), shape (height, width).

        Returns:
            volume: Transformed volume, shape (num_channels, num_slices, height, width).
        """
        if self.img_format != 'raw':
            raise NotImplementedError('Unsupported img_format: {}'.format(self.img_format))

        # Pad or crop to expected number of slices
        inputs = self._pad(inputs)

        if self.resize_shape is not None:
            inputs = self._rescale(inputs, interpolation=cv2.INTER_AREA)

        if self.crop_shape is not None:
            row_margin = max(0, inputs.shape[-2] - self.crop_shape[-2])
            col_margin = max(0, inputs.shape[-1] - self.crop_shape[-1])
            # Random crop during training, center crop during test inference
            row = random.randint(0, row_margin) if self.is_training_set else row_margin // 2
            col = random.randint(0, col_margin) if self.is_training_set else col_margin // 2
            inputs = self._crop(inputs, col, row, col + self.crop_shape[-1], row + self.crop_shape[-2])

        if self.do_vflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis=-2)

        if self.do_hflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis=-1)

        if self.do_rotate:
            angle = random.randint(-15, 15)
            inputs = rotate(inputs, angle, (-2, -1), reshape=False, cval=AIR_HU_VAL)

        # Normalize raw Hounsfield Units
        inputs = self._normalize_raw(inputs)

        inputs = np.expand_dims(inputs, axis=0)  # Add channel dimension
        inputs = torch.from_numpy(inputs)

        return inputs
    def contrast_volume(self, volume, lower=0.99, upper=1.01):
        """
        Apply a single random contrast factor to the entire volume.
        factor ∼ Uniform(lower, upper).
        Assumes input is already in [0,1]; output is clipped to [0,1].
        """
        factor = np.random.uniform(lower, upper)
        mean_intensity = volume.mean()
        vol = mean_intensity + factor * (volume - mean_intensity)
        return vol


    def brightness_volume(self, volume, lower=-0.01, upper=0.01):
        """
        Apply a single random brightness shift to the entire volume.
        shift ∼ Uniform(lower, upper) * intensity_range.
        Assumes input is in [0,1]; output is clipped to [0,1].
        """
        intensity_range = volume.max() - volume.min()
        shift = np.random.uniform(lower, upper) * intensity_range
        vol = volume + shift
        return vol

    def affine_volume_torch(self, volume, lower=0.8, upper=1.2, device='cuda'):
        """
        Apply a single random affine stretch to a 3D volume using PyTorch.
        - volume: np.ndarray shape (S, H, W), values assumed in [0,1]
        - lower/upper: range for scale factors
        - device: 'cuda' or 'cpu'
        
        Returns: np.ndarray shape (S, H, W) on CPU, same dtype float32.
        """
        # 1) Convert to a torch tensor and add batch+channel dims: (S,1,H,W)
        vol = torch.from_numpy(volume).unsqueeze(1).to(device, dtype=torch.float32)

        # 2) Sample one scale for x and y (same for every slice)
        scale_x = np.random.uniform(lower, upper)
        scale_y = np.random.uniform(lower, upper)

        # 3) Build the 2×3 affine transform matrix θ
        #    Here: [ [scale_x, 0, 0],
        #            [0, scale_y, 0] ]
        #    which scales the image around the center.
        theta = torch.tensor(
            [[scale_x, 0.0, 0.0],
            [0.0, scale_y, 0.0]],
            device=device,
            dtype=torch.float32
        ).unsqueeze(0)  # shape (1, 2, 3)

        # 4) Repeat θ for each slice in the batch dimension
        S, _, H, W = vol.shape
        theta = theta.repeat(S, 1, 1)  # (S, 2, 3)

        # 5) Create sampling grid and apply it in one go
        #    align_corners=False is the PyTorch default for newer versions
        grid = F.affine_grid(theta, size=vol.size(), align_corners=False)
        out = F.grid_sample(
            vol, grid,
            mode='bilinear',         # linear interpolation
            padding_mode='zeros',    # no reflection
            align_corners=False
        )

        # 6) Squeeze channel dim and bring back to CPU+NumPy
        return out.squeeze(1) 

    def _transform2(self, inputs):
        """Transform slices: resize, random crop, normalize, and convert to Torch Tensor.

        Args:
            inputs: 2D/3D NumPy array (un-normalized raw HU), shape (height, width).

        Returns:
            volume: Transformed volume, shape (num_channels, num_slices, height, width).
        """
        if self.img_format != 'raw':
            raise NotImplementedError('Unsupported img_format: {}'.format(self.img_format))

        # Pad or crop to expected number of slices
        inputs = self._pad(inputs)

        if self.resize_shape is not None:
            inputs = self._rescale(inputs, interpolation=cv2.INTER_AREA)
        if self.do_vflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis=-2)

        if self.do_hflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis=-1)

        if self.do_rotate:
            angle = random.randint(-15, 15)
            inputs = rotate(inputs, angle, (-2, -1), reshape=False, cval=AIR_HU_VAL) 
        
        if self.do_contrast:
            inputs = self.contrast_volume(inputs)
        if self.do_brightness:
            inputs = self.brightness_volume(inputs)
        if self.do_affine:
            inputs = self.affine_volume_torch(inputs)
        else:
            inputs = torch.from_numpy(inputs).to(torch.float32)
            
        if self.crop_shape is not None:
            # determine H, W
            if inputs.dim() == 2:
                H, W = inputs.shape
            else:  # (C, H, W)
                _, H, W = inputs.shape
            
            crop_h, crop_w = self.crop_shape[-2], self.crop_shape[-1]
            row_margin = max(0, H - crop_h)
            col_margin = max(0, W - crop_w)
        
        if self.is_training_set:
            # torch.randint high is exclusive, so add +1
            row = int(torch.randint(0, row_margin + 1, (1,)).item())
            col = int(torch.randint(0, col_margin + 1, (1,)).item())
        else:
            row = row_margin // 2
            col = col_margin // 2
        
        # crop: [row : row+crop_h, col : col+crop_w]
        if inputs.dim() == 2:
            inputs = inputs[row : row + crop_h,
                            col : col + crop_w]
        else:
            inputs = inputs[:, 
                            row : row + crop_h,
                            col : col + crop_w]
    
        # 2) Normalize raw Hounsfield Units (implement _normalize_raw to accept tensors)
        inputs = self._normalize_raw2(inputs)
        
        # 3) Add a channel dimension at front if needed
        # If inputs was (H, W), becomes (1, H, W); if (C, H, W), becomes (1, C, H, W)
        inputs = inputs.unsqueeze(0).cpu()
    # Add channel dimension
     

        
        # Order of transforms
        # Resize
        # Flip
        # Rotate
        # Bright
        # contrast
        # Convert to tensor
        # Affine
        # Crop
        # Normalize

        #all 
        #resize, flip, rotation
        #crop
         #norm
        return inputs

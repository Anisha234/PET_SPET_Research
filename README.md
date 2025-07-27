# A Two-Stage Chunk Aggregation Framework for Pulmonary Embolism Detection from CTPA Scans and Electronic Health Records
This respository contains the model and checkpoints described in the paper: A Two-Stage Chunk Aggregation Framework for Pulmonary Embolism Detection from CTPA Scans and Electronic Health Records 

## Abstract:
Pulmonary embolism (PE) is a life-threatening condition with a high mortality rate of 30%. Immediate treatment is crucial to improve treatment outcomes, but PE diagnosis often takes multiple days due to the limited availability of radiologists to analyze computed tomography pulmonary angiography (CTPA) images, which involves taking multiple X-ray images (often 100s of slices) of the chest region. Therefore, automation of diagnosis of PE can significantly improve patient outcomes. We use the RadFusion dataset to improve upon prior work to diagnose PE using the CTPA scans as well as using electronic health records. The labeling method and pre-processing of EHR data was improved, with over 50% of the features being identified as redundant. Correlation based analysis was done to select key EHR features, drastically reducing the number of features needed. A novel two stage pipeline was developed for analyzing CTPE images. This approach consists of a model to analyze a window of contiguous CTPE slices (Chunks), followed by another model to aggregate information across multiple chunks.  Proposed improvements to the sampling of chunks further boosted accuracy with the two stage pipeline for the images resulting in a 3% improvement in AUROC (0.95 compared to 0.92 for the PeNet model) with 1.8% improvement due to using a pre-trained DinoV2 small backbone, compared to a custom architecture and an additional 1.2% from improvements in aggregating information across chunks and improved sampling. Distilling the metadata features to just the top 16 most important features and using a random forest classifier resulted in the highest AUROC of 0.79 compared to the 0.76 when using all the metadata features. The analysis supports the results from larger datasets such as INSPECT and bridge the gap in prior work (RadFusion) which indicated that EHR data was more accurate than PE data.  We also demonstrate that modern self-supervised backbones trained on web scale data offer superior performance, reducing the need for custom architectures. This research also shows that very few EHR features contribute to accuracy, reducing the need for collection of large amounts of EHR data. The current approach isn't end-to-end trainable and separates chunk-level and patient-level models; in the future, we aim to explore unified models and more complex backbones.

Results of the S-PET Model

<img width="598" height="131" alt="image" src="https://github.com/user-attachments/assets/13c54648-4726-455f-96a7-c958c1ab45a9" />

## Usage 

### Download trained weights

The checkpoints and weights for SPET are https://github.com/Anisha234/ASSIP_research_penet/blob/master/sequence_model/best_model.pth respectively. 

### Training and Testing PET
To train the PET model, run train_dino_transformer.bat. To test this model, run test.bat after specifiying the right checkpoint. 


### Training SPET
To train the SPET model do the following:
1) Run test.bat with --window_shift = True and phase = train, phase = val, and phase = test to generate three sets of embeddings.
2) Repeat 1) with --window_shift = False.
3) Run sequence_train_model to train the model.
To test the SPET model, simply run test_sequence_model with the correct checkpoint. 



## Preprocessing the data
Change the input and output path of `/scripts/create_pe_hdf5_update.py` and run the script to generate dataset in hdf5 format for loading efficiency. This will create a `data.hdf5` file and put it under the same directory as the data.

## Acknowledgements
This repository is built of the PE-Net codebase. For more information, refer to https://github.com/marshuang80/penet/tree/master. 


python test.py --data_dir="C:\Users\preet\Documents\multimodalpulmonaryembolismdataset\hdf5" ^
                 --ckpt_path="C:\Users\preet\Documents\penet\train_logs\Test_20250715_134700\epoch_13.pth.tar" ^
                 --radfusion_np_path= "C:\\Users\\preet\\Documents\\multimodalpulmonaryembolismdataset\\all\\" ^
                 --results_dir=results ^
                 --phase=val ^
                 --crop_shape=256,256 ^
                 --name=first_test ^
                 --dataset=pe ^
                 --gpu_ids=0 ^
                 --num_workers=4 ^
                 --num_slices=24 ^
                 --batch_size=8 ^
                 --window_shift=False ^
                 
          
 
# weight-estimation-under-cover
Source code for our IJCARS paper [Seeing under the cover with a 3D U-Net: point cloud-based weight estimation of covered patients](https://link.springer.com/article/10.1007/s11548-021-02476-0).

## Dependencies
Please first install the following dependencies
* Python3 (we use 3.8.3)
* numpy
* pytorch (we tested 1.6.0 and 1.9.0)
* bps
* yacs
* scipy
* sklearn

## Data Preparation
1. Download the SLP dataset from https://web.northeastern.edu/ostadabbas/2019/06/27/multimodal-in-bed-pose-estimation/. Create a directory `/dataset/SLP` and move move the dataset to this directory. We recommend to create a symlink.
2. Execute `cd data` and `python preprocess_slp.py` to generate point clouds from the original depth images and to obtain the segmentation masks of uncovered patients. The data is written to `'/dataset/SLP/3d_data_{}_{}'.format(POSITION, COVER_COND)`.

## Training
1. In `/configs/defaults.py`, modify `_C.BASE_DIRECTORY` in line 5 to the root directory where you intend to save the results.
2. In the config files `/configs/CONFIG_TO_SPECIFY.yaml`, you can optionally modify `EXPERIMENT_NAME`in line 1. Models and log files will finally be written to `os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)`.
3. Navigate to the `main` directory
4. Execute `python train.py --gpu GPU --config-file ../configs/config_ours_uncovering.yaml --stage uncovering` to train the 3D U-Net to virtually uncover the patients. This corresponds to step 1 in our paper. After each epoch, we save the model weights and a log file to the specified directory.
5. Execute `python train.py --gpu GPU --config-file ../configs/config_ours_weight.yaml --stage weight` to train the 3D CNN for weight regression of patients that were previously uncovered by the 3D U-Net. This corresponds to step 2 in our paper. Again, we save the model weights and a log file to the specified directory after each epoch.

## Testing
* If you trained a model yourself following the instructions above, you can test the model by executing `python test.py --config-file ../configs/config_ours_weight.yaml --gpu GPU --val-split VAL_SPLIT --cover-condition COVER_COND --position POSITION`. `VAL_SPLIT`should be in {dana, sim}, where "dana" represents the lab setting and "sim" is the simulated hospital room. `COVER_COND` should be in {cover1, cover2, cover12}, and `POSITION` should be in {supine, lateral, all, left, right}. The output is the mean average error (MAE) in kg for specified setting, cover condition and patient position.
* Otherwise, we provide [pre-trained models](https://drive.google.com/drive/folders/1Bxw9qvqCxOL7YT55c_WtnsmvkTgQ9kp9). Download the models and use them for inference by executing `python test.py --config-file ../configs/config_ours_weight.yaml --gpu GPU --val-split VAL_SPLIT --cover-condition COVER_COND --position POSITION --unet-path /PATH/TO/UNET --cnn3d-path /PATH/TO/3DCNN`. These models achieve the following MAEs: supine & cover1: 4.61kg, lateral & cover1: 4.50kg, supine & cover2: 4.86kg, lateral & cover2: 4.53kg.

## Training and Testing of Baselines
1. To train one of the baseline models, execute `python train_baseline.py --gpu GPU --config-file ../configs/CONFIG_TO_SPECIFY.yaml` by specifying the desired config file.
2. For testing the trained baseline model, execute `python test_baseline.py --config-file ../configs/CONFIG_TO_SPECIFY.yaml --gpu GPU --val-split VAL_SPLIT --cover-condition COVER_COND --position POSITION`. Now, depending on the trained model, `COVER_COND` can / should be set to uncover as well.

## Citation
If you find our code useful for your work, please cocite the following paper
```latex
@article{bigalke2021seeing,
  title={Seeing under the cover with a 3D U-Net: point cloud-based weight estimation of covered patients},
  author={Bigalke, Alexander and Hansen, Lasse and Diesel, Jasper and Heinrich, Mattias P},
  journal={International journal of computer assisted radiology and surgery},
  year={2021},
  doi="10.1007/s11548-021-02476-0"
}
```


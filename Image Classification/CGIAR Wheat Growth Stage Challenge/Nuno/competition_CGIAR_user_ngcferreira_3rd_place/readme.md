Approach
----------------------------------------------------------------------------
1. Selected only high quality label images 

2. Created 5 Folds stratified by growth_stage

3. Trained 3 different models on the 5 folds.
    - Models:
        - SeResNeXt50_32x4d
        - EfficientNet B0
        - ResNet50
    
    - Used the following random augmentations in the train data:
        - Vertical Flip
        - Horizontal Flip
        - Rotation [-180, 180]
        - Blur
        - Coarse Dropout
        - Brightness and Contrast
        - Shift and Scale

    - Used a Gradual Warmup Scheduler, followed by a Cosine Annealing Learning Rate Scheduler
   
4 - Created pseudo labels for the test set, by combining (with equal weight) the output of each of the 
above models' predictions using a test time augmentation (TTA) with the following transforms:
- None
- Vertical Flip
- Horizontal Flip


5 - Appended the pseudo labels to training split of each of the 5 folds, 
and fine tuned the 3 models on this data for 20 epochs

6- Combined the prediction of the 3 fine tuned models, using again TTA with the following transforms:
- None
- Horizontal Flip

How to run it
----------------------------------------------------------------------------
- Pre-requirements:
    - Create conda environment with python 3.6:
        - conda create -n cgiar python=3.6
        
    - Activate environment
        - conda activate cgiar
        
    - Install all the required packages using the following command:
        - pip3 install -r requirements.txt
 
    - Expects raw data to be in a folder with the following structure:
        - <input_folder>/
            - Images/*.jpeg
            - Train.csv
            - SampleSubmission.csv

- Execute the following command and wait until all models are trained:
    - python3 train.py --data_folder <DATA_FOLDER_HERE> --output_folder <OUTPUT_FOLDER_HERE>
   
    - Example: 
        - python3 train.py --data_folder /cgiar/input --output_folder output_new

- A submission.csv is stored in current folder

- All models' checkpoints and intermediate predictions are also stored in the same output folder

- All code was run on a Linux Mint PC with
    - 64GB of RAM
    - a NVidia 1080 TI GPU (11GB RAM) 

- Total run-time of approx 29 hours to train all models
# Zindi Microsoft Rice Disease Classification Challenge
This solution is based on Imagenet pretrained Swin Transformer (https://arxiv.org/abs/2103.14030) model.

## Environment
Kaggle Notebooks or similar  
* Tesla P100 GPU
* CUDA Version: 11.0
* Cudatoolkit 10.2


## Directory structure
    .
    train.py 
    test.py
    models
    |── data/
    |  |── Images/
    |  |── Train.csv
    |  |── Test.csv
    |  |── SampleSubmission.csv


## To reproduce

### 1. Install dependencies
- `pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html`
- `pip install fastai==2.6.3 timm==0.5.4 `


### 2. Training

Run the training script for each fold. Total training time is ~5 hours. 
- `python train.py data 0`
- `python train.py data 1`
- `python train.py data 2`
- `python train.py data 3`
- `python train.py data 4`


Models will be saved to the `models` directory.

### 3. Inference

(Optional) Download pretrained models

`wget -O models.zip --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1v1cfRa4gO2qb560bRERw7z5OI4CU8qLT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1v1cfRa4gO2qb560bRERw7z5OI4CU8qLT" && rm -rf /tmp/cookies.txt`

`unzip -qqn models.zip`

Run the inference script. Inference time is ~10 minutes. Submission file will be saved to Submission/Submission.csv

- `python test.py data`  

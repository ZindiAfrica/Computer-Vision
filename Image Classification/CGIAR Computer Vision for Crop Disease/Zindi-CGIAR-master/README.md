# ICLR Workshop Challenge #1: CGIAR Computer Vision for Crop Disease - Submission description
This repository includes training and test scripts for the recognition system that scored 2nd place.

## Step By Step Guide

### Requirements

- Python 3.7
- numpy
- scipy
- Pillow
- tensorflow / tensorflow-gpu  (>=1.14.0)

You can use Dockerfile or requirements.txt file to Install Requirements.

If you want to use tensorflow-gpu follow https://www.tensorflow.org/install/gpu




### Preparation

#### Install requirements

##### Using Source

```bash
virtualenv -p /usr/bin/python3 CGIAR_ZINDI
pip3 install -r requirements.txt
source CGIAR_ZINDI/bin/activate
```

##### Using Docker
```bash
cd docker/cpu
docker build .
```

Currently it's not suitable for Conda. Might cause the Segmentation Fault.

#### Clone repository
git clone https://github.com/picekl/CGIAR.git

#### Download pre-trained weigths from 

Options:
1. https://github.com/tensorflow/models/tree/master/research/slim  (ImageNet Pre-trained)
2. http://ptak.felk.cvut.cz/personal/sulcmila/models/LifeCLEF2018/  <-- **Used while training**
3. http://ptak.felk.cvut.cz/personal/sulcmila/models/LifeCLEF2019/

#### Prepare CGIAR Data
1. Download CGIAR Data and Copy Paste them into data folder. Please follow 
```
data
└─── train_images
│    └─── healthy_wheat
│    │    └───AV7FQ8.jpg
│    └─── leaf_rust     
│    │    └───LVJTB0.jpg
│    └─── stem_rust
│         └───WKOFOG.jpg
└───train_images
    └───2KS7HG.jpg
```

Should look like follow: data/train_images/healthy_wheat etc.

#### Create training tf-records

1. Run read_labels.py script
```bash
python3 read_labels.py
```

This will generate metadata needed for tf_records.

2. Run build_tf_records.py
python3 build_tf_records.py


### Training

#### Update finetune_<model>_500.sh script

##### If using GPU Set for your visible device
export CUDA_VISIBLE_DEVICES=3

##### Path to the tf_recorts
TF_DATASET_DIR=/data/tf_records/

##### Path where the checkpoints are stored
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/CGIAR/checkpoints/inception_resnet_v2_plantclef_500_trainval_test_for_zindi/

##### Path to the pretrained Checkpoint -> DOWNLOAD FROM http://ptak.felk.cvut.cz/personal/sulcmila/models/LifeCLEF2019/
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/picekluk/pretrained_models/PlantCLEF2018/inception_resnet_v2/model.ckpt-1200000

##### Adjust your batch_size based on your ram capacity. (Smaller BS might affect results.)
--batch_size=16 


### Test-Time

1. Copy models to CGIAR/models
2. Run Evaluations:
python3 run.py
bash run.sh

**Make sure you have created a TF-Records out of test data!!**

### Pre-trained models

You can download pre-trained models from following link:

https://drive.google.com/open?id=1TqFEH37Ubs_8igkLVZ2fde8WKMtpT7r7

#### Usage - Pre-trained models

Copy 
- .meta
- .index
- .data

files for all the checkpoints/nets into CGIAR/models/. Follow "structure" bellow.

```
models
└─── inception_resnet_v2_500
│    └───model.ckpt-8000.data-00000-of-00001
│    └───model.ckpt-8000.index
│    └───model.ckpt-8000.meta
│    └───model.ckpt-18000.data-00000-of-00001
│    └───model.ckpt-18000.index
│    └───model.ckpt-18000.meta
└───inception_v4_500
    └───model.ckpt-4000.data-00000-of-00001
    └───model.ckpt-4000.index
    └───model.ckpt-4000.meta
```

### Trubleshooting
- https://www.tensorflow.org/install/gpu



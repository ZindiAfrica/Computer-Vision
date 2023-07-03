## Digital Africa Plantation Counting Challenge Solution

### 1. Hardware to reproduce

|Inferencing|Training|
|-----------|--------|
|>= 16GB RAM | A100 PCIe/SXM 40GB RAM|
|NVIDIA GPU  | NVIDIA GPU |

*Please 
use A100 if you want to reproduce the closest match to mine. I tried to reproduce this code on RTX 3090 and private score is 1,578 (got 1st). I think with this competition the hardware to reproduce is quite important.*

### 2. Install packages

```
sudo apt-get install libgl1-mesa-glx
conda env create -f environment.yml
conda activate zindi
```

### 3. Prepare dataset

Download all dataset from this link [zindi](https://zindi.africa/competitions/digital-africa-plantation-counting-challenge/data) and put it into folder [dataset](dataset).

unzip train images
```
cd dataset
unzip TreeImages.zip -d TreeImages
cd ..
```
Dataset structure should be:

```
|-- SampleSubmission.csv
|-- Test.csv
|-- Train.csv
|-- TreeImages
|   |-- Id_014krjcyn7.png
|   |-- Id_016a9g4o2g.png
|   |-- Id_01h0t20qbt.png
|   |-- Id_01myrj7ave.png
|   `-- Id_zzjon76p3b.png
`-- TreeImages.zip
```

### 4. Get started with inferencing
**For final structure of code please see structure.txt file**

I provide my train logs and my checkpoints for comparision.

Download logs and checkpoints at [link](https://drive.google.com/drive/folders/177gsnYAohHwgnVJViHHvayNxUE382JcY?usp=sharing)

Put all checkpoints into folder checkpoints and run

```
bash scripts/infer.sh
```
then
```
python ensemble_all.py
```
output of the submission will be shown in folder submission.

### 5. Training

```
bash scripts/train.sh
```
See output in folder logs and checkpoints. 

Training process will take about 4 hours for 6 models on A100.

### 6. Inferencing

```
bash scripts/infer.sh
```
This process will take about 1-2 minutes
Results of 6 models will be shown in folder results.

### 7. Submission

```
python ensemble_all.py
```

output of the submission will be shown in folder submission.



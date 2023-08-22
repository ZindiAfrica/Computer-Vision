# Rice disease : 5th place solution
## Solution summary
The solution consists of an ensemble of 9 models trained on the rgb data.
1. data preprocessing
* We deleted redundant images  in the rgb data.
   
*  create 5 stratified folds on  the Label column.
2. trained models
   

   
   
The proposed solution consists of an ensemble of 9 different models as described bellow:

|  conf name	| model type 	| model name  	|  img size	| 
|---	|---	|---	|---	|
|  beit_224aug	|  transformer	|  	beit_large_patch16_224| 224 	| 
| cnvxt_384 	| convnet 	|  convnext_large_384_in22ft1k	|  384	|  
|  deit_384	|  transformer	| deit3_base_patch16_384_in21ft1k 	|  384	|  
|  swin_base_256v2cv	|  transformer	|  swinv2_base_window12to16_192to256_22kft1k	|  256	| 
| swin_base_384v2 	|  transformer	|  swinv2_base_window12to24_192to384_22kft1k	| 384 	| 
|  swin_large_192v2	|  transformer	|  swinv2_large_window12_192_22k	|  192	|
|  vit_224_cv	|  transformer	|  vit_base_patch32_224_in21k	|  224	|
|  vit_384_cv	|  transformer	|  vit_base_patch32_384	|  384	|
|  vit_r50	|  hybrid	|  vit_base_r50_s16_384	|  384	|
1. Ensemble 


* ensemble with hill climbing  on the probabilities predictions

## Steps
1. Place competition data in `input/` directory
2. Run `notebooks\preprocess.ipynb` to create `input/train_rgb_purged.csv` containing purged data.
3. Run `./run.sh` to train all the models
4. Run the `notebooks\ensemble.ipynb`  to get the final submission file  
The output file is `ensemble_last.csv`

## Hardware/OS used for the competition
- CPU: AMD Ryzen™ 9 3900X, 24cores
- GPU: RTX 3080TI
- RAM: 32GB
- OS: Ubuntu 20.04

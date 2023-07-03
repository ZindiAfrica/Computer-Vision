# Digital Africa Plantation Counting Challenge

## How to run the code

### Steps

```
Run the notebooks in the following order:
```

* Run all the Training*.ipynb notebooks (TrainingConvnext, TrainingEffv2s, ...)  
  - Will save all the checkpoints into different folders in `models/`.   
  - Each folder has the following namimg: `{model_name}_{n_folds}_{image_size}_{lr}_{batch-size}_{experiment-id}`.
* Blending.ipynb
  - Infer all the models  
  - Generate the final subsmission file (`submissions/submission_wmm.csv`)

## [On the Leaderboard](https://zindi.africa/competitions/digital-africa-plantation-counting-challenge/leaderboard/teams/wemovemountains)

Rank : 4th   
Evaluation score:   
&nbsp;- Metric        : RMSE (lower is better)   
&nbsp;- Public score  : 2.030653631      
&nbsp;- Private score : 1.581690205   

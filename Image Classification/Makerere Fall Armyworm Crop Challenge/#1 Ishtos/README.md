[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# Makerere-Fall-Armyworm-Crop-Challenge-Solution

## Requirement

`pip install -r requirements.txt`

## Run

```bash
cd src/exp_003
```

### Train

Train models. The weights are stored in `exp_003/outputs/checkoints` directory.
Please move the original weights I provide if necessary.

```bash
./run_train.sh
```

### Validation

Validate using trained models. The oof files are stored in `exp_003/outputs/` directory.
The original oof files I provide is in `exp_003/outputs/csvs` directory.

```bash
./run_valid.sh
```

### Test

Test using trained models. The prediction files are stored in `exp_003/outputs/` directory.
The original prediction files I provide is in `exp_003/outputs/csvs` directory.

```bash
./run_train.sh
```

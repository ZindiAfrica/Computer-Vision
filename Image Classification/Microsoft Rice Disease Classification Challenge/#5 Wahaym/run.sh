
PYTHONPATH=. python src/train.py --config_path config/beit_224aug.yaml --exp_name beit_224aug  --save_checkpoint

PYTHONPATH=. python src/train.py --config_path config/deit_384.yaml --exp_name deit_384  --save_checkpoint

PYTHONPATH=. python src/train.py --config_path config/swin_base_384v2.yaml --exp_name swin_base_384v2  --save_checkpoint

PYTHONPATH=. python src/train.py --config_path config/swin_large_192v2.yaml --exp_name swin_large_192v2  --save_checkpoint

PYTHONPATH=. python src/train.py --config_path config/swin_base_256v2cv.yaml --exp_name swin_base_256v2cv  --save_checkpoint

PYTHONPATH=. python src/train.py --config_path config/vit_r50.yaml --exp_name vit_r50  --save_checkpoint

PYTHONPATH=. python src/train.py --config_path config/cnvxt_384.yaml --exp_name cnvxt_384  --save_checkpoint

PYTHONPATH=. python src/train.py --config_path config/vit_384_cv.yaml --exp_name vit_384_cv  --save_checkpoint

PYTHONPATH=. python src/train.py --config_path config/vit_224_cv.yaml --exp_name vit_224_cv  --save_checkpoint

CUDA_VISIBLE_DEVICES=0 python train.py -c configs/tf_efficientnetv2_s.yaml
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/tf_efficientnetv2_l_in21k.yaml
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/tf_efficientnetv2_l.yaml
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/tf_efficientnetv2_m_in21ft1k.yaml
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/tf_efficientnetv2_m_in21k.yaml
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/tf_efficientnetv2_m.yaml
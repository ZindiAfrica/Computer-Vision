import os


class SeResNextConfig:
    root_dir = '/media/nuno/barracuda/cgiar-wheat-rust/input'
    output_dir = 'outputs'
    images_dir = os.path.join(root_dir, 'Images')
    regression = True

    folds = 5
    image_size = 224
    train_batch_size = 64
    val_batch_size = 128
    num_workers = 4
    num_classes = 1 if regression else 7

    neural_backbone = 'seresnext50_32x4d'
    init_lr = 1e-4
    epochs = 300
    dropout = 0.3

    use_gradual_lr_warmup = True
    warmup_factor = 10
    warmup_epo = 1
    learning_rate_factor = 0.8
    learning_rate_patience = 5


class Resnet50Config:
    root_dir = '/media/nuno/barracuda/cgiar-wheat-rust/input'
    output_dir = 'outputs'
    images_dir = os.path.join(root_dir, 'Images')
    regression = True

    folds = 5
    image_size = 224
    train_batch_size = 100
    val_batch_size = 256
    num_workers = 4
    num_classes = 1 if regression else 7

    neural_backbone = 'resnet50'
    init_lr = 1e-4
    epochs = 300
    dropout = 0.2

    use_gradual_lr_warmup = True
    warmup_factor = 10
    warmup_epo = 1
    learning_rate_factor = 0.8
    learning_rate_patience = 5


class EfficientNetB0:
    root_dir = '/media/nuno/barracuda/cgiar-wheat-rust/input'
    output_dir = 'outputs'
    images_dir = os.path.join(root_dir, 'Images')
    regression = True

    folds = 5
    image_size = 512
    train_batch_size = 15
    val_batch_size = 16
    num_workers = 4
    num_classes = 1 if regression else 7

    neural_backbone = 'efficientnet_b0'
    init_lr = 1e-4
    epochs = 300
    dropout = 0.3

    use_gradual_lr_warmup = True
    warmup_factor = 10
    warmup_epo = 1
    learning_rate_factor = 0.8
    learning_rate_patience = 5



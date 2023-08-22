# Utils
import argparse
import glob
from pathlib import Path
import yaml
import pandas as pd
# Deep learning Stuff
from torch.utils.data import DataLoader
import ttach as tta

# Function Created by me
from dataset import *
from model import *
from train_func import *


def main(cfg):
    test_df = pd.read_csv(cfg['test_file'])
    probabilitys = None
    seed_everything(cfg['seed'])
    gc.enable()
    device = return_device()
    test_df['file_path'] = test_df['Image_id'].apply(lambda x: return_filpath(x, folder=cfg['test_dir']))

    test_dataset = Cultivar_data_inference(image_path=test_df['file_path'],
                                           cfg=cfg,
                                           transform=get_train_transforms(cfg['image_size']),
                                           transform_rgn=get_train_transforms_rgn(cfg['image_size']))

    test_loader = DataLoader(
        test_dataset, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
    )

    for path in glob.glob(f"{cfg['model_path']}/*.pth"):
        model = BaseModelFeature(cfg)
        model.load_state_dict(torch.load(path))
        model = tta.ClassificationTTAWrapper(model, tta.aliases.flip_transform())

        model.to(device)
        model.eval()
        probablity = inference_fn(test_loader, model, cfg)

        if probabilitys is None:
            probabilitys = probablity / 5
        else:
            probabilitys += probablity / 5
        del model
        gc.collect()
        torch.cuda.empty_cache()
    blast = []
    brown = []
    healthy = []
    probabilitys = probabilitys.detach().cpu().numpy()
    for i in probabilitys:
        blast.append(i[0])
        brown.append(i[1])
        healthy.append(i[2])
    sub = pd.DataFrame({"filename": test_df['Image_id'], "blast": blast, "brown": brown, "healthy": healthy})
    np.save(cfg['probablity_file'], probabilitys, allow_pickle=True)
    sub.to_csv(cfg['submission_file'], index=False)


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    with open(str(args.file), "r") as stream:
        config = yaml.safe_load(stream)
    main(config)

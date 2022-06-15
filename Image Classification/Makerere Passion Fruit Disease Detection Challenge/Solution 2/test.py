import pandas as pd
import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import argparse
import os


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./mmdetection/configs/detectors/customized_detectorRs2.py", help='config file')
    parser.add_argument('--work_dir', type=str, default='work_dir1/detectors/fold3', help='batch size')
    parser.add_argument('--fold', type=int, default=3, help='fold')
    parser.add_argument('--root', type=str, default='.', help='root')
    parser.add_argument('--checkpoint', type=str, default='./work_dir1/detectors/fold3/epoch_17.pth', help='checkpoint')
    parser.add_argument('--device_ids', type=int, default=3, help='gpu number')
    parser.add_argument('--test_csv', type=str, default='my_test.csv', help='save test csv')
    return parser.parse_args()


def test(outputs):
    res = []
    class_dict = {
        0: 'fruit_healthy',
        1: 'fruit_woodiness',
        2: 'fruit_brownspot'

    }
    for id, bboxes in enumerate(outputs):
        image_id = test_df.iloc[id]['Image_ID']
        for idx, bbox in enumerate(bboxes):
            if len(bbox) != 0:
                for single_bbox in bbox:
                    xmin, ymin, xmax, ymax, probability = single_bbox[0], single_bbox[1], single_bbox[2], single_bbox[
                        3], single_bbox[4]
                    # xmax = xmin + width
                    # ymax = ymin + height

                    my_dict = {
                        'Image_ID': image_id,
                        'class': class_dict[idx],
                        'confidence': probability,
                        'ymin': ymin,
                        'xmin': xmin,
                        'ymax': ymax,
                        'xmax': xmax
                    }
                    res.append(my_dict)

            # print(bbox)
    return res


def predict(test_df):
    args = argparser()
    print(args.config)
    cfg = Config.fromfile(args.config)
    cfg.work_dir = os.path.join(args.root, args.work_dir)
    mmcv.mkdir_or_exist(cfg.work_dir)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    print('dataset has been build..')
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=2,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint)  # , map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model, device_ids=[3])
    outputs = single_gpu_test(model, data_loader)
    res = test(outputs, test_df)

    my_test_df = pd.DataFrame(res, columns=['Image_ID', 'class', 'confidence', 'ymin', 'xmin', 'ymax', 'xmax'])
    my_test_df.to_csv(f'{args.test_csv}', index=False)


if __name__ == "__main__":
    test_df = pd.read_csv("./Test.csv")
    predict(test_df)


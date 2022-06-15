import cv2
import ast

import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as immg

import random

import torch
import os
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from tqdm.notebook import tqdm
import json
import warnings
warnings.filterwarnings("ignore")
#from ensemble_boxes import *
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


def create_result_df(outputs, test_df):
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
    my_test_df = pd.DataFrame(res, columns=['Image_ID', 'class', 'confidence', 'ymin', 'xmin', 'ymax', 'xmax'])
    return my_test_df


def get_max_4(test4, csv_name = 'result.csv'):
    visited = []

    frames = []
    for index, row in test4.iterrows():
        if row['Image_ID'] not in visited:
            visited.append(row['Image_ID'])
            if len(test4.loc[test4.Image_ID == row['Image_ID']]) <= 4:
                frames.append(test4.loc[test4.Image_ID == row['Image_ID']])
            else:
                frames.append(test4.loc[test4.Image_ID == row['Image_ID']].sort_values(by=['confidence'], ascending=False)[:4])

    result = pd.concat(frames)
    result.to_csv(csv_name, index = False)





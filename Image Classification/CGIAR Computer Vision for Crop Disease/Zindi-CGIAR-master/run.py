#!/usr/bin/env python

import numpy as np
import traceback
import glob

from scipy.stats.mstats import gmean

from CGIAR.slim.eval_image_classifier import eval_images

# Note : List of train-classes
CLASSES = ['leaf_rust', 'stem_rust', 'healthy_wheat']


def prepare_test_data():

    file_paths = []
    file_names = []
    texts = []
    labels = []

    images_paths = glob.glob('data/test_images/*.jpg') + glob.glob('data/test_images/*.JPG') + glob.glob('data/test_images/*.jfif')

    for image_path in images_paths:
        file_names.append(image_path.split("/")[-1])
        file_paths.append(image_path)
        texts.append(str(0))
        labels.append(int(0))

    #images_to_tfrecords(file_paths, texts, labels)

    return file_names


def predict(prediction_info_list, file_names):

    for prediction_info in prediction_info_list:
        print('Prediction over:', prediction_info)
        tta, model_info = prediction_info

        eval_images(num_images=len(file_names), iteration=model_info[-1],
                    central_fraction=tta[0], mirror=tta[1],
                    rotation=tta[2], model_info=model_info)

        prediction_file = open('CGIAR/testing_dir/' + model_info + '.csv', 'w')
        prediction_file.write("ID,leaf_rust,stem_rust,healthy_wheat\n")

        ids = []
        fc_values_list = np.load('CGIAR/testing_dir/' + model_info + '.fcvalues.npy')
        filename_to_fc_values = {}

        with open('CGIAR/testing_dir/' + model_info + '.filenames') as image_names:
            for fc_values, filename in zip(fc_values_list, image_names):
                filename = filename.strip()

                if filename in ids:
                    continue
                else:
                    ids.append(filename)
                    filename_to_fc_values[filename] = fc_values

        for filename in file_names:
            predictions = filename_to_fc_values[filename]

            # index = np.argmax(predictions)
            # if predictions[index] > 0.5:
            #     predictions[index] = 1.0

            PREDICTION_LINE = "%s,%1.8f,%1.8f,%1.8f\n" % (
            filename.split(".")[0], predictions[1], predictions[2], predictions[0])
            prediction_file.write(PREDICTION_LINE)  # Write output

        prediction_file.close()


def combine_predictions(prediction_info_list):

    predictions = {}

    for prediction_info in prediction_info_list:

        tta, model_info = prediction_info

        csv_path = 'CGIAR/testing_dir/' + model_info + '.csv'

        with open(csv_path, mode='r') as csv_file:
            for row in csv_file:
                id, leaf_rust, stem_rust, healthy_wheat = row.split(',')

                if id == 'ID':
                    continue
                if id not in predictions:
                    predictions[id] = [[float(leaf_rust), float(stem_rust), float(healthy_wheat.strip())]]
                else:
                    predictions[id].append([float(leaf_rust), float(stem_rust), float(healthy_wheat.strip())])

    # output_path = 'inception_v4_from_PlantClef2018_500/inception_v4_from_PlantClef2018_500-model.ckpt' + \
    #               '-4000_0.7+0.8+0.9+1.0_all_flip_gmean_no_sharpening___' + \
    #               'inception_resnet_v2_plantclef_500_trainval-model.ckpt-8000_0.8+0.8flip_no_sharpening' + \
    #               '18000_0.8+0.8flip_no_sharpening.csv'


    output_path = 'reimplemented_best_.csv'

    with open(output_path, 'w') as fo:
        fo.write("ID,leaf_rust,stem_rust,healthy_wheat\n")
        for fname, predictions in predictions.items():
            # predictions = np.mean(predictions, axis=0)
            predictions = gmean(predictions, axis=0)

            fid = fname.split('.')[0]
            line = "%s,%1.8f,%1.8f,%1.8f\n" % (fname, predictions[0], predictions[1], predictions[2])
            fo.write(line)


def run():

    prediction_info_list = [[[0.8, False, False], 'CGIAR/models/inception_v4_500/model.ckpt-4000_0'],
                            [[0.9, False, False], 'CGIAR/models/inception_v4_500/model.ckpt-4000_1'],
                            [[1.0, False, False], 'CGIAR/models/inception_v4_500/model.ckpt-4000_2'],
                            [[0.8, True, False], 'CGIAR/models/inception_v4_500/model.ckpt-4000_3'],
                            [[0.9, True, False], 'CGIAR/models/inception_v4_500/model.ckpt-4000_4'],
                            [[1.0, True, False], 'CGIAR/models/inception_v4_500/model.ckpt-4000_5'],
                            #[[0.7, False, False], 'CGIAR/models/inception_v4_500/model.ckpt-4000_6'],
                            #[[0.7, True, False], 'CGIAR/models/inception_v4_500/model.ckpt-4000_7'],
                            [[0.8, False, False], 'CGIAR/models/inception_resnet_v2_500/model.ckpt-8000_0'],
                            #[[0.9, False, False], 'CGIAR/models/inception_resnet_v2_500/model.ckpt-8000_1'],
                            #[[1.0, False, False], 'CGIAR/models/inception_resnet_v2_500/model.ckpt-8000_2'],
                            [[0.8, True, False], 'CGIAR/models/inception_resnet_v2_500/model.ckpt-8000_3'],
                            #[[0.9, True, False], 'CGIAR/models/inception_resnet_v2_500/model.ckpt-8000_4'],
                            #[[1.0, True, False], 'CGIAR/models/inception_resnet_v2_500/model.ckpt-8000_5'],
                            [[0.8, False, False], 'CGIAR/models/inception_resnet_v2_500/model.ckpt-18000_0'],
                            #[[0.9, False, False], 'CGIAR/models/inception_resnet_v2_500/model.ckpt-18000_1'],
                            #[[1.0, False, False], 'CGIAR/models/inception_resnet_v2_500/model.ckpt-18000_2'],
                            [[0.8, True, False], 'CGIAR/models/inception_resnet_v2_500/model.ckpt-18000_3'],
                            #[[0.9, True, False], 'CGIAR/models/inception_resnet_v2_500/model.ckpt-18000_4'],
                            #[[1.0, True, False], 'CGIAR/models/inception_resnet_v2_500/model.ckpt-18000_5']
                             ]

    file_names = prepare_test_data()
    print('TF - Records Created')
    predict(prediction_info_list, file_names)
    print('Predictions Done')
    combine_predictions(prediction_info_list)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        error = traceback.format_exc()
        print(error)

# 2019, Lukas Picek, based on a previous script for tensorflow.
# Original license below:
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from CGIAR.slim.datasets import dataset_factory
from CGIAR.slim.nets import nets_factory
from CGIAR.slim.preprocessing import preprocessing_factory
from CGIAR.slim.my_slim import evaluation as slim_evaluation

import os
import numpy as np

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 20, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'CGIAR', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', 'data/tf_records', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'inception_v4', 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.999,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 500, 'Eval image size')

tf.app.flags.DEFINE_boolean(
    'save_prediction_list', True, 'Save the list of all predictions to eval_dir.')

tf.app.flags.DEFINE_boolean(
    'save_logits_list', False, 'Save the 2D list of all logits to eval_dir.')

tf.app.flags.DEFINE_string(
    'save_fc', 'Predictions', 'Save a numpy list of FC layer values to eval_dir.')

tf.app.flags.DEFINE_boolean(
    'save_filenames', True, 'Save a txt list of evaluated image filenames.')

tf.app.flags.DEFINE_integer(
    'num_augmentations', 0, 'If "num_augmentations">0, test n-times with random augmentations ')

########################
# Session Config Flags #
########################
tf.app.flags.DEFINE_boolean(
    'modest', False,
    'A "modest" run only consumes necessary GPU memory ~ session_config.gpu_options.allow_growth=True.')

FLAGS = tf.app.flags.FLAGS


def eval_images(num_images, iteration='1', central_fraction=0.9, mirror=None, rotation=None, model_info=None):
    import os
    print(os.getcwd())

    if 'v4' in model_info:
        eval_dir = 'CGIAR/testing_dir/inception_v4_500'
        checkpoint_path = 'CGIAR/models/inception_v4_500/model.ckpt-4000'
        model_name = 'inception_v4'
    else:
        eval_dir = 'CGIAR/testing_dir/inception_resnet_v2_500'
        model_name = 'inception_resnet_v2'
        if '18000' in model_info:
            checkpoint_path = 'CGIAR/models/inception_resnet_v2_500/model.ckpt-18000'
        else:
            checkpoint_path = 'CGIAR/models/inception_resnet_v2_500/model.ckpt-8000'


    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        num_classes = dataset.num_classes - FLAGS.labels_offset

        network_fn = nets_factory.get_network_fn(
            model_name,
            num_classes=num_classes,
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        if FLAGS.save_filenames:
            [image, label, filename] = provider.get(['image', 'label', 'filename'])
        else:
            [image, label] = provider.get(['image', 'label'])

        label -= FLAGS.labels_offset

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or model_name
        # perform training-like augmentations if FLAGS.num_augmentations>0
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=(FLAGS.num_augmentations > 0))

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size, central_fraction=central_fraction, mirror=mirror, rotation=rotation)

        if FLAGS.save_filenames:
            images, labels, filenames = tf.train.batch(
                [image, label, filename],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)
        else:
            images, labels = tf.train.batch(
                [image, label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        logits, end_points = network_fn(images)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall@3': slim.metrics.streaming_sparse_recall_at_k(logits, labels, 3),
        })

        # Note: slim.metrics.streaming_recall_at_k replaced with slim.metrics.streaming_sparse_recall_at_k

        names_to_updates_ordered = [names_to_updates['Accuracy'], names_to_updates['Recall@3']]

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        num_batches = math.ceil(num_images / float(FLAGS.batch_size))

        if FLAGS.num_augmentations > 0:
            num_batches *= FLAGS.num_augmentations

        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        else:
            checkpoint_path = checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)

        if FLAGS.modest:
            # Configure session to take only needed GPU memory
            print("[CONFIG] Using limited GPU resources and CPU threads.")
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = True
            session_config.intra_op_parallelism_threads = 1  # start only one CPU thread !
            session_config.inter_op_parallelism_threads = 1  # start only one CPU thread !
        else:
            session_config = None

        eval_ops = [names_to_updates['Accuracy'], names_to_updates['Recall@3']]
        return_eval_ops = [0, 1]

        if FLAGS.save_prediction_list:
            eval_ops.append(predictions)
        if FLAGS.save_logits_list:
            eval_ops.append(logits)
            print("Logits variable :: ", logits)
        if FLAGS.save_fc is not None:
            fc = end_points[FLAGS.save_fc]
            eval_ops.append(fc)
        if FLAGS.save_filenames:
            eval_ops.append(filenames)

        final_op_value, eval_op_values = slim_evaluation.evaluate_once(
            master=FLAGS.master,
            checkpoint_path=checkpoint_path,
            logdir=eval_dir,
            num_evals=num_batches,
            eval_op=eval_ops,
            final_op=names_to_updates_ordered,
            session_config=session_config,
            variables_to_restore=variables_to_restore)

        additional_suffix = ''

        if FLAGS.save_prediction_list:
            predictions_file = os.path.join(eval_dir,
                                            os.path.basename(checkpoint_path) + additional_suffix + '_' + iteration + ".predictions")
            print("Saving predictions to ", predictions_file)
            final_predictions = eval_op_values.pop(2)
            with open(predictions_file, 'w') as f:
                for batch in final_predictions:
                    for item in batch:
                        f.write("%d\n" % item)
        if FLAGS.save_logits_list:
            logits_file = os.path.join(eval_dir,
                                       os.path.basename(checkpoint_path) + additional_suffix + '_' + iteration + ".logits")
            print("Saving logits to ", logits_file)
            final_logits = eval_op_values.pop(2)
            with open(logits_file, 'w') as f:
                for batch in final_logits:
                    for item in batch:
                        f.write(str(item.tolist())[1:-1] + "\n")
        if FLAGS.save_fc:
            fc_values = np.array(eval_op_values.pop(2))
            fc_values = fc_values.reshape(
                [fc_values.shape[0] * fc_values.shape[1], fc_values.shape[2]])  # join batches together
            fc_file = os.path.join(eval_dir,
                                   os.path.basename(checkpoint_path) + additional_suffix + '_' + iteration + ".fcvalues.npy")
            print("Saving FC layer values to ", fc_file)
            np.save(fc_file, fc_values)

        if FLAGS.save_filenames:
            filenames_file = os.path.join(eval_dir,
                                          os.path.basename(checkpoint_path) + additional_suffix + '_' + iteration + ".filenames")
            print("Saving image filenames to ", filenames_file)
            final_filenames = eval_op_values.pop(2)
            # print(final_filenames)
            with open(filenames_file, 'w') as f:
                for batch in final_filenames:
                    for item in batch:
                        f.write(item.decode("utf-8") + "\n")



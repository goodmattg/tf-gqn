"""
Test script to check the data input pipeline.
"""

import os
import sys

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
TF_GQN_ROOT = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
sys.path.append(TF_GQN_ROOT)

import tensorflow as tf
import numpy as np

from data_provider.gqn_provider import EagerDataReader


# constants
DATASET_ROOT_PATH = os.path.join(TF_GQN_ROOT, "data")
DATASET_NAME = "rooms_ring_camera"
CTX_SIZE = 5  # number of context (image, pose) pairs for a given query pose
BATCH_SIZE = 2

data_reader = EagerDataReader(
    DATASET_ROOT_PATH,
    DATASET_NAME,
    CTX_SIZE,
    mode=tf.estimator.ModeKeys.TRAIN,
    batch_size=BATCH_SIZE,
)

# Pull a single sample from the dataset
sample = data_reader.dataset.take(1)

for (features, labels) in sample:
    # print shapes of fetched objects
    print("Shapes of fetched tensors:")
    print("Query camera poses: %s" % str(features.query_camera.shape))
    print("Target images: %s" % str(labels.shape))
    print("Context camera poses: %s" % str(features.context.cameras.shape))
    print("Context frames: %s" % str(features.context.frames.shape))

print("TEST PASSED!")

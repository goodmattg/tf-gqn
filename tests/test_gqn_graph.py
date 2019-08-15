"""
Quick test script to shape-check graph definition of full GQN model with random
toy data.
"""

import os
import sys

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
TF_GQN_ROOT = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(TF_GQN_ROOT)

import tensorflow as tf
import numpy as np

from gqn_v2.gqn_params import GQN_DEFAULT_CONFIG
from gqn_v2.gqn_graph import gqn_draw, gqn_vae
from data_provider.gqn_provider import EagerDataReader

# constants
_BATCH_SIZE = 1
_CONTEXT_SIZE = GQN_DEFAULT_CONFIG.CONTEXT_SIZE
_DIM_POSE = GQN_DEFAULT_CONFIG.POSE_CHANNELS
_DIM_H_IMG = GQN_DEFAULT_CONFIG.IMG_HEIGHT
_DIM_W_IMG = GQN_DEFAULT_CONFIG.IMG_WIDTH
_DIM_C_IMG = GQN_DEFAULT_CONFIG.IMG_CHANNELS
_SEQ_LENGTH = GQN_DEFAULT_CONFIG.SEQ_LENGTH

# constants
DATASET_ROOT_PATH = os.path.join(TF_GQN_ROOT, "data")
DATASET_NAME = "rooms_ring_camera"
CTX_SIZE = 5  # number of context (image, pose) pairs for a given query pose
BATCH_SIZE = 1

data_reader = EagerDataReader(
    DATASET_ROOT_PATH,
    DATASET_NAME,
    CTX_SIZE,
    mode=tf.estimator.ModeKeys.TRAIN,
    batch_size=BATCH_SIZE,
)

# Pull a single sample from the dataset
sample = data_reader.dataset.take(1)

features, labels = next(sample.__iter__())

# feed single batch input through the graph
query_pose = features.query_camera
target_frame = labels
context_poses = features.context.cameras
context_frames = features.context.frames

# graph definition
net, ep_gqn = gqn_draw(
    query_pose=query_pose,
    target_frame=target_frame,
    context_poses=context_poses,
    context_frames=context_frames,
    model_params=GQN_DEFAULT_CONFIG,
    is_training=True,
)

mu = net

print(mu)
print(mu.shape)

for ep, t in ep_gqn.items():
    print(ep, t.shape)

print("TEST PASSED!")

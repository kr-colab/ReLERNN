import glob
import pickle
import sys
import msprime as msp
import numpy as np
import os
import multiprocessing as mp
import shlex
import shutil
import random
import copy
import argparse
import h5py
import allel
import time

from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras import optimizers
from keras.optimizers import RMSprop
from keras.models import Model,Sequential,model_from_json
from keras import layers
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, Conv1D, MaxPooling2D, AveragePooling2D,concatenate, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

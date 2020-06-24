from enum import Enum

class Update_T(Enum):
  FULL = 1
  PARTIAL = 2
  RPARTIAL = 3
  HALF = 4
  RHALF = 5
# enum for random vector generator type
class Generator_T(Enum):
  Vanilla = 1
  Baklava = 2

class Kernel_T(Enum):
  DOT = 0
  COS = 1
  PL1 = 2 # This is binary - polarize as in {(-1, 1)}
  PL3 = 3
  PL4 = 4
  BT1 = 5 # This is binary as in {(0,1)} with hamming distance
  BT3 = 6
  BT4 = 7

mapping = { Kernel_T.BT1: [0, 1],
            Kernel_T.BT3: [6.19, 11.7, 35.3, 92.3, 189, 303, 410, 499],
            Kernel_T.BT4: [1.02, 1.44, 3.10, 7.21, 16.3, 34.1, 65.9, 115,
                           180, 254, 329, 398, 459, 513, 558, 598]}

################ Data #####################

# DATA set: its dir and filename.
DATA = {
  "is": ("isolet", "isolet"),
  "shs": ("smart_home", "smart_home_split"),
  "es": ("ExtraSensory", "pos"),
  "mn": ("MNIST", "MNIST"),
  "uc": ("UCIHAR", "UCIHAR"),
  "vt": ("votes", "votes")
}

config = {
  "data_location" : "../dataset/",     # Location for all the data
  "directory"     : DATA["uc"][0],
  "dataset"       : DATA["uc"][1],   # directory and dataset

  ################ HD general #####################
  # Dimension of HD vectors
  "D" : 2000,
  # Gaussian random vector generation
  "vector" : "Gaussian",  # Gaussian
  "mu" : 0,
  "sigma" : 1,
  # binary vector
  "binarize" : 0,
  # Learning rate
  # if binarize make lr 1
  #"lr" : 0.037,
  "lr" : 1,
  # Obsolete: whether the vector should be sparse, and how sparse
  "sparse" : 0,
  "s" : 0.1,
  # binary model
  "binaryModel" : 0,
  "checkpoints": False, # whether to have checkpoint files.
  "kernel": Kernel_T.COS,

  ################### Baklava #######################
  "width": None,
  "height": None,
  # Number of layers for the Baklava
  "nLayers" : 5,
  # Whether the dimensions for the layers are uniform
  "uniform_dim" : 1,
  # Whether the filter/kernel sizes for the layers are uniform
  "uniform_ker" : 1,

  # Dimensions for each layers (non-uniform layer); preferably sums up to D
  # If uniform_dim = 1, then d = D // nLayers
  "dArr" : None,

  # Filter/kernel size for every layer (uniform filter); preferably, k | width-1 and height-1 of 2d features.
  "k" : 3,
  # Filter sizes for each layer (non-uniform filter); each preferably divides width-1 and height-1
  "kArr" : None,

  ################### One-shot learning ###############
  # Master switch
  "one_shot": 0,
  # the percentage of data to actually use (for automation)
  "data_percentages": [1.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
  # default rate
  "train_percent": 1,


  ################## Dropout ##########################
  # Master switch
  "dropout": 0,
  # dropout rate during each period; 0 means no dropout (for automation)
  "drop_percentages": [0, 0.1, 0.2, 0.5],
  # default rate
  "dropout_rate": 0,
  "update_type": Update_T.FULL,

  ################## Train / Test iterations ##########
  # number of trials to run per experiment
  "iter_per_trial": 3,
  # number of times to run per encoding
  "iter_per_encoding": 5,
  # iterations per training (number of epochs)
  "epochs": 100,
}

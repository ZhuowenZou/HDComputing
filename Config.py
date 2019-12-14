from enum import Enum

class Update_T(Enum):
  FULL = 1
  PARTIAL = 2
  RPARTIAL = 3

################ Data #####################

# DATA set: its dir and filename.
DATA = {
  "is": ("isolet", "isolet"),
  "sh": ("smart_home", "smart_home_split"),
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
  "D" : 2500,
  # Gaussian random vector generation
  "vector" : "Gaussian",  # Gaussian
  "mu" : 0,
  "sigma" : 1,
  # binary vector
  "binarize" : 0,
  # Learning rate
  # if binarize make lr 1
  "lr" : 0.037, #"lr" : 1,
  # Obsolete: whether the vector should be sparse, and how sparse
  "sparse" : 0,
  "s" : 0.1,
  # binary model
  "binaryModel" : 0,

  ################### Baklava #######################
  # Number of layers for the Baklava
  "nLayers" : 5,
  # Whether the dimensions for the layers are uniform
  "uniform_dim" : 1,
  # Whether the filter/kernel sizes for the layers are uniform
  "uniform_ker" : 1,

  # Dimension for every layer (uniform layer); preferably divides D
  "d" : None,
  # Dimensions for each layers (non-uniform layer); preferably sums up to D
  "dArr" : None,

  # Filter/kernel size for every layer (uniform filter); preferably, k | width and height of 2d features and k^2 | d.
  "k" : None,
  # Filter sizes for each layer (non-uniform filter); each preferably divides width and height and corresponding dArr[i]
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
  "epochs": 300,
}

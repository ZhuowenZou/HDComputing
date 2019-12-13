import sys
import struct
import numpy as np
from Config import config

# Dataloader: load data from designated location. Default location is Config
# train_param: dictionary for training parameters, including nFeatures, nClasses, data(X), label(y)
# test_param: dictionary for test param.
# nFeatures: number of features in the dataset
# nClasses: number of classes in the dataset

class Dataloader:

    # load in data and populate dataloader
    def __init__(self, dir=config["directory"], dataset=config["dataset"], data_loc=config["data_location"]):

        data_path = data_loc + dir + "/" + dataset

        sys.stderr.write("Loading train data...\n")
        self.train_param = self.readChoirData(data_path, "train")

        sys.stderr.write("Loading test data...\n")
        self.test_param = self.readChoirData(data_path, "test")

        self.nFeatures = self.train_param["nFeatures"]
        self.nClasses = self.train_param["nClasses"]

        sys.stderr.write("Data Loaded. Num of features = "+str(self.nFeatures)+" Num of Classes = "+str(self.nClasses))

    # read in data from file
    def readChoirData(self, data_path, data_type):

        filename = data_path + "_" + data_type + ".choir_dat"
        param = dict()

        with open(filename, 'rb') as f:
            nFeatures = struct.unpack('i', f.read(4))[0]
            nClasses = struct.unpack('i', f.read(4))[0]
            X = []
            y = []
            while True:
                newDP = []
                for i in range(nFeatures):
                    v_in_bytes = f.read(4)
                    if v_in_bytes is None or len(v_in_bytes) == 0:
                        param["nFeatures"], param["nClasses"], param["data"], param["labels"] = \
                            nFeatures, nClasses, np.asarray(X), np.asarray(y)
                        return param
                    v = struct.unpack('f', v_in_bytes)[0]
                    newDP.append(v)
                l = struct.unpack('i', f.read(4))[0]
                X.append(newDP)
                y.append(l)

    def getTrain(self):
        return self.train_param["X"], self.train_param["y"]

    def getTest(self):
        return self.test_param["X"], self.test_param["y"]

    def getParam(self):
        return self.nFeatures, self.nClasses

import Config
import sys
import time
import math
import random
import numpy as np
from Config import config
import joblib
from enum import Enum

class Update_T(Enum):
  FULL = 1
  PARTIAL = 2
  RPARTIAL = 3
  HALF = 4

def sgn(i):
  if i > 0:
    return 1
  else:
    return -1

# n = e^-(|x|^2/(2std^2))
def gauss(x,y,std):
  n = np.linalg.norm(x - y)
  n = n ** 2
  n = n * -1
  n = n / (2 * (std**2))
  n = np.exp(n)
  return n

def poly(x,y,c,d):
  return (np.dot(x,y) + c) ** d

#  dot product/ gauss product/ cos product
def kernel(x,y):
  dotKernel = np.dot
  gaussKernel = lambda x, y : gauss(x,y,25)
  polyKernel = lambda x,y : poly(x,y,3,5)
  cosKernel = lambda x,y : np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))
  #k = gaussKernel
  #k = polyKernel
  k = dotKernel
  #k = cosKernel
  return k(x,y)

class HD_classifier:

    # Required parameters for the training it supports; will enhance later
    options = ["one_shot", "dropout", "lr"]
    # required opts for dropout
    options_dropout = ["dropout_rate", "update_type"]

    # id: id associated with the basis/encoded data
    def __init__(self, D, nClasses, id):
        self.D = D
        self.nClasses = nClasses
        self.classes = np.zeros((nClasses, D))
        # If first fit, print out complete configuration
        self.first_fit = True
        self.id = id

    def resetModel(self, basis = None, D = None, nClasses = None, id = None, reset = True):
        if basis is not None:
            self.basis = basis
        if D is not None:
            self.D = D
        if nClasses is not None:
            self.nClasses = nClasses
        if id is not None:
            self.id = id
        if reset:
            self.resetClasses()
        self.first_fit = True

    def resetClasses(self):
        self.classes = np.zeros((self.nClasses, self.D))

    def getClasses(self):
        return self.classes

    def update(self, weight, mask, guess, answer, rate, update_type=Update_T.FULL):
        sample = weight * mask
        if update_type == Update_T.FULL:
            self.classes[guess]  -= rate * sample
            self.classes[answer] += rate * sample
        elif update_type == Update_T.PARTIAL:
            self.classes[guess]  -= rate * sample
            self.classes[answer] += rate * weight
        elif update_type == Update_T.RPARTIAL:
            self.classes[guess]  -= rate * weight
            self.classes[answer] += rate * sample
        elif update_type == Update_T.HALF:
            self.classes[answer] += rate * weight
        else:
            raise Exception("unrecognized Update_T")

    # update class vectors with each sample, once
    # return train accuracy
    def fit(self, data, label, param = None):

        assert self.D == data.shape[1]

        # Default parameter
        if param is None:
            param = Config.config
        for option in self.options:
            if option not in param:
                param[option] = config[option]
        if self.first_fit:
            sys.stderr.write("Fitting with configuration: %s \n" % str([(k,param[k]) for k in self.options]))

        # Actual fitting

        # handling dropout
        mask = np.asarray([1 for _ in range(self.D)])
        if param["dropout"]:
            for option in self.options_dropout:
                if option not in param:
                    param[option] = config[option]
            # Mask for dropout
            for i in np.random.choice(self.D, int(self.D * (param["drop_rate"])), replace=False):
                mask[i] = 0

        # fit
        r = list(range(data.shape[0]))
        random.shuffle(r)
        correct = 0
        count = 0
        for i in r:
            sample = data[i] * mask
            assert data[i].shape == mask.shape

            answer = label[i]
            maxVal = -1
            guess = -1
            for m in range(self.nClasses):
                val = kernel(self.classes[m], sample)
                if val > maxVal:
                    maxVal = val
                    guess = m
            if guess != answer:
                self.update(data[i], mask, guess, answer, param["lr"])
            else:
                correct += 1
            count += 1
        self.first_fit = False
        return correct / count

    def test(self, data, label):

        assert self.D == data.shape[1]

        # fit
        r = list(range(data.shape[0]))
        random.shuffle(r)
        correct = 0
        count = 0
        for i in r:
            answer = label[i]
            maxVal = -1
            guess = -1
            for m in range(self.nClasses):
                val = kernel(self.classes[m], data[i])
                if val > maxVal:
                    maxVal = val
                    guess = m
            if guess == answer:
                correct += 1
            count += 1
        return correct / count


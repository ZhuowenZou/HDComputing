import Config
import sys
import random
import numpy as np
from Config import config, Update_T, Kernel_T


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
def kernel(x,y, kernel_t = Kernel_T.COS):
  dotKernel = np.dot
  gaussKernel = lambda x, y : gauss(x,y,25)
  polyKernel = lambda x,y : poly(x,y,3,5)
  cosKernel = lambda x,y : np.dot(x,y) / (np.linalg.norm(x))

  if kernel_t == Kernel_T.COS:
      k = cosKernel
  elif kernel_t == Kernel_T.DOT:
      k = dotKernel
  elif kernel_t == Kernel_T.Bin:
      print("TODO_TYPES!")
      k = None
  else:
      print("Type unrecognized!")
      k = None
  return k(x, y)

# X: set of vectors; y: one vector
def batch_kernel(X, y, kernel_t = Kernel_T.COS):
    dotKernel = lambda X, y: np.matmul(y, X.T)
    cosKernel = lambda X, y: np.matmul(y, X.T)/(np.linalg.norm(X, axis = 1))
    if kernel_t == Kernel_T.COS:
        k = cosKernel
    elif kernel_t == Kernel_T.DOT:
        k = dotKernel
    elif kernel_t == Kernel_T.Bin:
        # remember that when dealling with hamming distance, the smaller the better
        print("TODO_TYPES!")
        k = None
    else:
        print("Type unrecognized!")
        k = None
    return k(X, y)

class HD_classifier:

    # Required parameters for the training it supports; will enhance later
    options = ["one_shot", "dropout", "lr", "kernel"]
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
            self.classes[guess]  -= rate * weight
            self.classes[answer] += rate * weight
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
        mask = np.ones(self.D)
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
            #maxVal = -1
            #guess = -1
            #for m in range(self.nClasses):
            #    val = kernel(self.classes[m], sample)
            #    if val > maxVal:
            #        maxVal = val
            #        guess = m
            vals = batch_kernel(self.classes, sample, param["kernel"])
            guess = np.argmax(vals)
            
            if guess != answer:
                self.update(data[i], mask, guess, answer, param["lr"])
            else:
                correct += 1
            count += 1
        self.first_fit = False
        return correct / count

    # Used for one-pass training. Adaptive learning (learning each sample with weight) will be added in the future.
    # fit_type currently only support None, which is naive update
    def fit_once(self, data, label, param = None, fit_type = None):

        assert self.D == data.shape[1]

        # Default parameter
        if param is None:
            param = Config.config
        for option in self.options:
            if option not in param:
                param[option] = config[option]
        if self.first_fit:
            sys.stderr.write("Fitting with configuration: %s \n" % str([(k, param[k]) for k in self.options]))

        # Actual fitting

        # fit
        r = list(range(data.shape[0]))
        random.shuffle(r)
        correct = 0
        count = 0
        for i in r:
            sample = data[i]
            answer = label[i]

            #maxVal = -1
            #guess = -1
            #for m in range(self.nClasses):
            #    val = kernel(self.classes[m], sample)
            #    if val > maxVal:
            #        maxVal = val
            #        guess = m
            vals = batch_kernel(self.classes, sample)
            guess = np.argmax(vals)

            self.update(sample, np.ones(self.D), guess, answer, param["lr"], Update_T.HALF)

            if guess == answer:
                correct += 1
            count += 1
        self.first_fit = False
        return correct / count

    def predict(self, data):

        assert self.D == data.shape[1]

        prediction = []

        # fit
        for i in range(len(data.shape[0])):
            maxVal = -1
            guess = -1
            for m in range(self.nClasses):
                val = kernel(self.classes[m], data[i])
                if val > maxVal:
                    maxVal = val
                    guess = m
            prediction.append(guess)
        return prediction

    # TODO: reduce this to using predict??
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


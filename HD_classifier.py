import Config
import sys
import random
import numpy as np
import sklearn
from Config import config, Update_T


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
        self.counts = np.zeros(nClasses)
        # If first fit, print out complete configuration
        self.first_fit = True
        self.id = id
        self.idx_weights = np.ones((D))
        self.update_cnts = np.zeros((D))
        self.mask = np.ones((D))

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
        self.counts[guess] += 1
        self.counts[answer] += 1
        if update_type == Update_T.FULL:
            self.classes[guess]  -= rate * weight
            self.classes[answer] += rate * weight
        elif update_type == Update_T.PARTIAL:
            self.classes[guess]  -= rate * sample
            self.classes[answer] += rate * weight
        elif update_type == Update_T.RPARTIAL:
            self.classes[guess]  -= rate * weight
            self.classes[answer] += rate * sample
        elif update_type == Update_T.MASKED:
            self.classes[guess]  -= rate * sample
            self.classes[answer] += rate * sample
        elif update_type == Update_T.HALF:
            self.classes[answer] += rate * weight
            self.counts[guess] -= 1
        elif update_type == Update_T.WEIGHTED:
            self.classes[guess]  -= rate * np.multiply(self.idx_weights, sample)
            self.classes[answer] += rate * np.multiply(self.idx_weights, sample)
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
        #if self.first_fit:
        #    sys.stderr.write("Fitting with configuration: %s \n" % str([(k,param[k]) for k in self.options]))

        # Actual fitting

        # handling dropout
        mask = np.ones(self.D)
        if param["masked"]:
            mask = np.copy(self.mask)
        elif param["dropout"]:
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
            vals = np.matmul(sample, self.classes.T)
            guess = np.argmax(vals)
            
            if guess != answer:
                self.update(data[i], mask, guess, answer, param["lr"], param["update_type"])
            else:
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

    # given current classifier value, return:
    # Variance of each dimension across the classes, and
    # The indices in the order from least variance to greatest
    def evaluateBasis(self):
        #normed_classes = self.classes/(np.sqrt(np.asarray([self.counts])).T)
        #variances = np.var(self.classes, axis = 0)
        normed_classes = sklearn.preprocessing.normalize(np.asarray(self.classes), norm='l2')
        variances = np.var(normed_classes, axis = 0) 
        assert len(variances) == self.D
        order = np.argsort(variances)
        return variances, order

    # Some basis are to be update
    def updateClasses(self, toChange = None):
        if toChange is None:
            #self.classes = np.zeros((self.nClasses, self.D))
            self.classes = sklearn.preprocessing.normalize(np.asarray(self.classes), norm='l2', axis = 0)
            self.counts = np.ones(self.nClasses) # An averaged vector is already in
        else:
            for i in toChange:
                self.classes[:,i] = np.zeros(self.nClasses)
    
    #Update update rates
    def updateWeights(self, toChange):
        #new_weight = max(self.idx_weights) + 1
        self.idx_weights = self.idx_weights/2
        for i in toChange:
            #self.idx_weights[i] = new_weight
            #self.idx_weights[i] += 1
            self.idx_weights[i] = 1
            self.update_cnts[i] += 1

    def updateMask(self, toChange):
        self.mask = np.ones((self.D))
        np.put(self.mask, toChange, 0, mode = "raise")


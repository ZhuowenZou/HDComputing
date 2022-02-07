import Config
import sys
import random
import numpy as np
import copy
import sklearn
from scipy import stats
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

def binarize(X):
    return np.where(X >= 0, 1, -1)

# X should be the class matrix of shape nClasses * D
# 0 is mapped to most lowest value and 2^(bits)-1 highest
def quantize(X, bits):
    Nbins = 2**bits
    # ultimate cheess
    bins = [ (i/(Nbins)) for i in range(Nbins)]
   # notice the axis along which to normalize is always the last one
    nX = stats.norm.cdf(stats.zscore(X, axis = X.ndim-1))
    nX = np.digitize(nX, bins) - 1
    #print("Max and min bin value:", np.max(nX), np.min(nX))
    #print("Quantized from ", X)
    #print("To", nX)
    return nX


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
  elif kernel_t == Kernel_T.PL1:
      print("TODO_TYPES!")
      k = None
  else:
      print("Type unrecognized!")
      k = None
  return k(x, y)

def hamming_map(X, y, mapping):
    ans = []
    for x in X:
        vec = abs(x - y)
        #print(x ,y, vec)
        vec = [mapping[int(v)] for v in vec]
        ans.append(sum(vec))
    ans = max(ans) - np.asarray(ans) # invert distance so that larger is closer
    return ans

# X: set of vectors; y: one vector
def batch_kernel(X, y, kernel_t = Kernel_T.COS):
    #print("Kernel type:", kernel_t)
    ny = copy.deepcopy(y)
    dotKernel = lambda X, y: np.matmul(y, X.T)
    cosKernel = lambda X, y: np.matmul(y, X.T)/(np.linalg.norm(X, axis = 1))
    if kernel_t == Kernel_T.COS:
        k = cosKernel
    elif kernel_t == Kernel_T.DOT:
        k = dotKernel
    elif kernel_t in [Kernel_T.PL1, Kernel_T.PL3, Kernel_T.PL4]:
        # remember that when dealing with hamming distance, the smaller the better
        k = dotKernel
    elif kernel_t in [Kernel_T.BT1, Kernel_T.BT3, Kernel_T.BT4]:
        k = lambda X, y: hamming_map(X,y, Config.mapping[kernel_t])
    else:
        print("Unrecognized batch kernel type! revert to dot")
        k = dotKernel
    return k(X, ny)

class HD_classifier:

    # Required parameters for the training it supports; will enhance later
    options = ["one_shot", "dropout", "lr", "kernel"]
    # required opts for dropout
    options_dropout = ["dropout_rate", "update_type"]

    # id: id associated with the basis/encoded data
    def __init__(self, D, nClasses, id):
        self.D = D
        self.nClasses = nClasses
        #self.classes = np.zeros((nClasses, D))
        self.classes = np.random.normal(0, 0.01, (nClasses, D))
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


    def preprocess_classes(self, X, kernel_t):
        # Make copy of inference model and data (for quantization)
        # Preprocess inference model according to eval method
        new_X = None
        if kernel_t == Kernel_T.PL1:
            new_X = binarize(X)
        elif kernel_t == Kernel_T.PL3:
            new_X = quantize(X, 3) - (3 ** 2 - 1) / 2
        elif kernel_t == Kernel_T.PL4:
            new_X = quantize(X, 4) - (4 ** 2 - 1) / 2
        elif kernel_t == Kernel_T.BT1:
            new_X = quantize(X, 1)
        elif kernel_t == Kernel_T.BT3:
            new_X = quantize(X, 3)
        elif kernel_t == Kernel_T.BT4:
            new_X = quantize(X, 4)
        else:
            new_X = copy.deepcopy(X)
        return np.asarray(new_X)

    def preprocess_data(self, X, kernel_t):
        new_X = None
        if kernel_t == Kernel_T.BT1:
            new_X = quantize(X, 1)
        elif kernel_t == Kernel_T.BT3:
            new_X = quantize(X, 3)
        elif kernel_t == Kernel_T.BT4:
            new_X = quantize(X, 4)
        else:
            new_X = copy.deepcopy(X)
        return np.asarray(new_X)

    # sep fitting updates class vectors only after all evaluation is done
    # Support binary, 1/3/4-bit training
    def fit_sep(self, data, label, param = Config.config):

        assert self.D == data.shape[1]

        # Default parameter
        for option in self.options:
            if option not in param:
                param[option] = config[option]
        if self.first_fit:
            sys.stderr.write("Sep Fitting with configuration: %s \n" % str([(k, param[k]) for k in self.options]))

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

        # Make copy of inference model and data (for quantization)
        kernel_t = param["kernel"]
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        data_cp = self.preprocess_data(data, kernel_t)

        # fit
        r = list(range(data.shape[0]))
        random.shuffle(r)
        correct = 0
        count = 0
        for i in r:
            assert data[i].shape == mask.shape

            sample = data_cp[i] * mask
            answer = label[i]
            # maxVal = -1
            # guess = -1
            # for m in range(self.nClasses):
            #    val = kernel(self.classes[m], sample)
            #    if val > maxVal:
            #        maxVal = val
            #        guess = m
            vals = batch_kernel(og_classes, sample, param["kernel"])
            guess = np.argmax(vals)

            if guess != answer:
                self.update(data[i], mask, guess, answer, param["lr"])
            else:
                correct += 1
            count += 1
        self.first_fit = False
        return correct / count

    # update class vectors with each sample, once
    # return train accuracy
    # TODO: not yet support BT1 BT3 BT4
    def fit(self, data, label, param = Config.config):

        assert self.D == data.shape[1]

        # Default parameter
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
    def fit_once(self, data, label, param = Config.config):

        assert self.D == data.shape[1]

        # Default parameter
        for option in self.options:
            if option not in param:
                param[option] = config[option]
        if self.first_fit:
            sys.stderr.write("Fitting with configuration: %s \n" % str([(k, param[k]) for k in self.options]))

        # Actual fitting

        # fit
        r = list(range(data.shape[0]))
        random.shuffle(r)
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
            guess = 0 # guess is a don't care

            self.update(sample, np.ones(self.D), guess, answer, param["lr"], Update_T.HALF)

        self.first_fit = False
        return -1

    def predict(self, data, param = Config.config):

        assert self.D == data.shape[1]

        prediction = []

        # Make copy of inference model and data (for quantization)
        kernel_t = param["kernel"]
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        data_cp = self.preprocess_data(data, kernel_t)

        for i in range(data.shape[0]):
            vals = batch_kernel(og_classes, data_cp[i], kernel_t)
            guess = np.argmax(vals)
            prediction.append(guess)
        return prediction

    def test(self, data, label, kernel_t = Config.config["kernel"]):

        assert self.D == data.shape[1]

        # Make copy of inference model and data (for quantization)
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        data_cp = self.preprocess_data(data, kernel_t)

        # fit
        correct = 0
        count = 0
        for i in range(data.shape[0]):
            answer = label[i]
            vals = batch_kernel(og_classes, data_cp[i], kernel_t)
            guess = np.argmax(vals)
            if guess == answer:
                correct += 1
            count += 1
        return correct / count

    def test(self, data, label, kernel_t = Config.config["kernel"]):

        assert self.D == data.shape[1]

        # Make copy of inference model and data (for quantization)
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        data_cp = self.preprocess_data(data, kernel_t)

        # fit
        correct = 0
        count = 0
        for i in range(data.shape[0]):
            answer = label[i]
            vals = batch_kernel(og_classes, data_cp[i], kernel_t)
            guess = np.argmax(vals)
            if guess == answer:
                correct += 1
            count += 1
        return correct / count

    def test_with_error(self, data, label, err_rate, err_lvl, kernel_t = Config.config["kernel"]):

        def gen_err(classes, err_rate, err_lvl, kernel_t):
            amountFlip = int(self.D * err_rate)
            new_classes = []
            for curr_class in classes:
                print("Class given:\n", curr_class)

                indices = [random.randint(0, self.D - 1) for _ in range(amountFlip)]
                signes = [2 * random.randint(0, 1) - 1 for _ in range(amountFlip)]
                for i in range(amountFlip):
                    ub = 0 # Upper bound
                    if kernel_t == Kernel_T.BT3:
                        ub = 7
                    elif kernel_t == Kernel_T.BT4:
                        ub = 15
                    else:
                        print("Kernel_T in error testing not supported!")
                        return None
                    temp = curr_class[indices[i]] + signes[i] * err_lvl
                    #temp = curr_class[indices[i]] + err_lvl
                    curr_class[indices[i]] = min(ub, max(0, temp)) # Upper Lower bound
                print("Class returned:\n", curr_class)
                new_classes.append(curr_class)
            return np.asarray(new_classes)

        assert self.D == data.shape[1]

        # Make copy of inference model and data (for quantization)
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        #print("From: ", og_classes[0])
        og_classes = gen_err(og_classes, err_rate, err_lvl, kernel_t)
        #print("To  : ", og_classes[0])
        data_cp = self.preprocess_data(data, kernel_t)

        # fit
        correct = 0
        count = 0
        for i in range(data.shape[0]):
            answer = label[i]
            vals = batch_kernel(og_classes, data_cp[i], kernel_t)
            guess = np.argmax(vals)
            if guess == answer:
                correct += 1
            count += 1
        return correct / count


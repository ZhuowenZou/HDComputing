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
def kernel(x,y, kernel_t = Config.config["kernel"]):
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
def batch_kernel(X, y, kernel_t = Config.config["kernel"]):
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
        self.classes = np.random.normal(0, 0.037, (nClasses, D))
        # If first fit, print out complete configuration
        self.first_fit = True
        self.id = id
        self.mask2d = np.zeros((nClasses, nClasses, D))
        self.bmask2d = None
        self.mask1d = None
        self.bmask1d = None
        self.decider = np.zeros((nClasses, nClasses))
        self.debug = False

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

    def normalizeClasses(self):
        self.classes = sklearn.preprocessing.normalize(np.asarray(self.classes), norm='l2')

    def update(self, weight, answer, guess, rate, mask = None, update_type=Update_T.FULL):
        sample = weight
        if mask is not None:
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
        #data_cp = self.preprocess_data(data, kernel_t)

        # fit
        r = list(range(data.shape[0]))
        random.shuffle(r)
        correct = 0
        count = 0
        for i in r:
            assert data[i].shape == mask.shape

            sample = data[i] * mask
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
                self.update(data[i], answer, guess, param["lr"], mask)
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
                self.update(data[i], answer, guess, param["lr"], mask)
            else:
                correct += 1
            count += 1
        self.first_fit = False
        return correct / count

    def fit_mask(self, data, label, param = Config.config):

        assert self.D == data.shape[1]
        # Default parameter
        for option in self.options:
            if option not in param:
                param[option] = config[option]
        if self.first_fit:
            sys.stderr.write("Fitting with configuration: %s \n" % str([(k, param[k]) for k in self.options]))

        # fit
        r = list(range(data.shape[0]))
        random.shuffle(r)
        correct = 0
        count = 0
        for i in r:
            sample = data[i]

            answer = label[i]
            vals = batch_kernel(self.classes, sample, param["kernel"])
            order = list(reversed(np.argsort(vals)))
            guesses = order[:2] # HARD-CODE TOP TWO

            if (guesses[0] != answer) and (answer in guesses):
                self.update(data[i], answer, guesses[0], param["lr"], None)
                # Mask update only update directionally because this won't lose information
                self.mask2d[answer][guesses[0]] += data[i] * (self.classes[answer] - self.classes[guesses[0]])
            elif (guesses[0] != answer) and (answer not in guesses):
                self.update(data[i], answer, guesses[0], param["lr"], None)
            else:
                correct += 1
                #self.mask2d[answer][guesses[0]] += data[i] * (self.classes[answer] - self.classes[guesses[1]])
            count += 1
        self.first_fit = False
        return correct / count

    def make_mask(self, data, label, dominance = None, param = Config.config):

        self.mask2d = np.zeros((self.nClasses, self.nClasses, self.D))

        assert self.D == data.shape[1]
        # Default parameter
        for option in self.options:
            if option not in param:
                param[option] = config[option]
        if self.first_fit:
            sys.stderr.write("Fitting with configuration: %s \n" % str([(k, param[k]) for k in self.options]))

        # fit
        r = list(range(data.shape[0]))
        random.shuffle(r)
        for i in r:
            sample = data[i]
            answer = label[i]
            vals = batch_kernel(self.classes, sample, param["kernel"])
            order = list(reversed(np.argsort(vals)))
            fst, snd = order[:2] # HARD-CODE TOP TWO
            fst_sc = vals[fst]
            snd_sc = vals[snd]
            ratio = fst_sc/snd_sc
            if answer in [fst, snd] and dominance is not None and ratio >= dominance:
                #print("Dominaed")
                useless = 1
            elif answer == fst:
                #print("Updating mask (%d, %d)"%(fst, snd))
                #self.mask2d[fst][snd] += abs(data[i] * (self.classes[fst] - self.classes[snd]))
                self.mask2d[fst][snd] += data[i] * (self.classes[fst] - self.classes[snd])
                pass
            elif answer == snd:
                self.mask2d[snd][fst] += data[i] * (self.classes[snd] - self.classes[fst])
        return

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

            self.update(sample, answer, guess, param["lr"], np.ones(self.D), Update_T.HALF)

        self.first_fit = False
        return -1


    def predict(self, data, kernel_t = Config.config["kernel"]):

        assert self.D == data.shape[1]

        prediction = []

        # Make copy of inference model and data (for quantization)
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        #data_cp = self.preprocess_data(data, kernel_t)

        for i in range(data.shape[0]):
            vals = batch_kernel(og_classes, data[i], kernel_t)
            guess = np.argmax(vals)
            prediction.append(guess)
        return prediction

    def predict_topn(self, data, topn = 1, kernel_t = Config.config["kernel"]):

        assert self.D == data.shape[1]

        prediction = []

        # Make copy of inference model and data (for quantization)
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        #data_cp = self.preprocess_data(data, kernel_t)

        for i in range(data.shape[0]):
            vals = batch_kernel(og_classes, data[i],kernel_t)
            order = list(reversed(np.argsort(vals)))
            guesses = order[:topn]
            prediction.append(guesses)
        return prediction


    def test(self, data, label, kernel_t = Config.config["kernel"]):

        assert self.D == data.shape[1]

        # Make copy of inference model and data (for quantization)
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        #data_cp = self.preprocess_data(data, kernel_t)

        # fit
        correct = 0
        count = 0
        for i in range(data.shape[0]):
            answer = label[i]
            vals = batch_kernel(og_classes, data[i], kernel_t)
            guess = np.argmax(vals)
            if guess == answer:
                correct += 1
            count += 1
        return correct / count

    def test_topn(self, data, label, topn = 1, kernel_t = Config.config["kernel"]):

        assert self.D == data.shape[1]

        # Make copy of inference model and data (for quantization)
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        #data_cp = self.preprocess_data(data, kernel_t)

        # fit
        correct = 0
        count = 0
        for i in range(data.shape[0]):
            answer = label[i]
            vals = batch_kernel(og_classes, data[i], kernel_t)
            order = list(reversed(np.argsort(vals)))
            guesses = order[:topn]
            if answer in guesses:
                correct += 1
            count += 1
        return correct / count


    def prep_bi_mask(self, c1_mask, c2_mask, percent, debug = None):

        if debug is None:
            debug = self.debug

        mask = c1_mask - c2_mask
        bmask = np.zeros(len(mask))
        amountKeep = int(percent * len(mask))
        # print(mask)

        order = np.argsort(mask)
        if debug:
            print("The negative of these values are assigned to -1")
            print(mask[order[:amountKeep]])
            print("These are assigned to 1")
            print(mask[order[-amountKeep:]])
        for k in order[:amountKeep]:
            if mask[k] < 0:
                bmask[k] = 0
        for k in order[-amountKeep:]:
            if mask[k] > 0:
                bmask[k] = 1
        return bmask


    def prep_mask(self, percent = 0.25):

        # Normalize mask2d
        masks = self.mask2d = np.asarray(self.mask2d)
        #masks = self.mask2d = np.reshape(sklearn.preprocessing.normalize(np.reshape(self.mask2d, (-1, self.D)),
        #                                                                 norm='l2'),
        #                                 (self.nClasses, self.nClasses, self.D))
        bmask2d = np.zeros(masks.shape)
        amountKeep = int(self.D * percent)
        #print(amountKeep)

        # 2d polarized mask
        for i in range(self.nClasses):
            for j in range(self.nClasses):
                # Skip the mask if is nan
                if np.isnan(np.sum(masks[i][j])) or np.isnan(np.sum(masks[j][i])) or i >= j:
                    continue
                #order = np.argsort(masks[i][j])
                #for k in order[:amountKeep]:
                #    bmask2d[i][j][k] = -1
                #for k in order[-amountKeep:]:
                #    bmask2d[i][j][k] = 1
                bmask2d[i][j] = self.prep_bi_mask(masks[i][j], masks[j][i], percent)
                bmask2d[j][i] = -bmask2d[i][j]
        self.bmask2d = bmask2d

        mask1d = np.zeros((self.nClasses, self.D))
        # 1d original mask
        for i in range(self.nClasses):
            for j in range(self.nClasses):
                if i != j and not np.isnan(np.sum(masks[i][j])):
                    mask1d[i] += masks[i][j]
                    # mask1d[i] -= masks[j][i]
        self.mask1d = mask1d

        bmask1d = np.zeros(mask1d.shape)
        for i in range(len(masks)):
            # Skip the mask if is nan
            if np.isnan(np.sum(mask1d[i])):
                continue
            order = np.argsort(mask1d[i])
            for k in order[:amountKeep]:
                bmask1d[i][k] = -1
            for k in order[-amountKeep:]:
                bmask1d[i][k] = 1
        self.bmask1d = bmask1d
        return


    def set_decider(self, data, label, threshold = 0.5, dominance = None, mask_t = "o", mask_d = 2, kernel_t = Config.config["kernel"], debug = None):

        if debug is None:
            debug = self.debug
        
        if debug:
            print("set_decider invokes test_mask. You may ignore output for now")
            
        _, _, _, _, _, _, _, _, mat, net_mat = self.test_mask(data, label, 0, dominance, mask_t, mask_d, kernel_t, debug)
        rate = net_mat/mat
        
        if debug:
            print("Before patitition:\n", rate)
        for i in range(self.nClasses):
            for j in range(self.nClasses):
                if np.isnan(rate[i][j]):
                    rate[i][j] = 0
                elif rate[i][j] <= threshold:
                    rate[i][j] = -1
                elif rate[i][j] >= 1-threshold:
                    rate[i][j] = 1
                else:
                    rate[i][j] = 0
        if debug:
            print("After patitition:\n", rate)
        self.decider = rate
        return rate

    def mask_selector(self, mask_t, mask_d):
        masks = None
        if mask_t == "o":
            if mask_d == 1:
                masks = self.mask1d
            elif mask_d == 2:
                masks = self.mask2d
        elif mask_t == "b":
            if mask_d == 1:
                masks = self.bmask1d
            elif mask_d == 2:
                masks = self.bmask2d
        return masks

    def test_mask(self, data, label, threshold = 1, dominance = None, mask_t = "o", mask_d = 2, kernel_t = Config.config["kernel"], debug = None):

        if debug is None:
            debug = self.debug
        assert self.D == data.shape[1]

        # Make copy of inference model and data (for quantization)
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        #data_cp = self.preprocess_data(data, kernel_t)

        # fit
        correct = 0
        correct_1 = 0
        correct_2 = 0
        correct_r = 0

        wrong_1 = 0
        wrong_2 = 0
        wrong_3 = 0 # When the answer is not in top 2

        matrix = np.zeros((self.nClasses, self.nClasses))
        net_matrix = np.zeros((self.nClasses,self.nClasses))
        count = 0

        masks = self.mask_selector(mask_t, mask_d)
        if masks is None:
            print("OOF NO MASK OF THE TYPE FOUND")
            return None

        for i in range(data.shape[0]):
            answer = label[i]
            vals = batch_kernel(og_classes, data[i], kernel_t)
            order = list(reversed(np.argsort(vals)))
            fst = order[0]
            snd = order[1]
            guess = fst
            the_mask = None
            fst_sc = vals[fst]
            snd_sc = vals[snd]
            ratio = fst_sc/snd_sc
            if dominance is not None and ratio >= dominance:
                guess = fst
            elif not np.isnan(np.sum(masks[fst][snd])) and not np.isnan(np.sum(masks[snd][fst])):
                if mask_d == 2:
                    the_mask = masks[fst][snd] - masks[snd][fst]
                    #if np.sum(the_mask) == 0:
                    #    print("Found untrained mask (%d, %d) during testing" % (fst, snd))
                    #    print("Their submasks starts with:")
                    #    print(masks[fst][snd][0:10], masks[snd][fst][0:10])
                    #    print("Selecting fst by default")
                    #    continue
                    fst_sc = np.dot((data[i] * the_mask), self.classes[fst])
                    snd_sc = -np.dot((data[i] * the_mask), self.classes[snd])
                    if snd_sc > fst_sc:
                        guess = snd
                    #score = np.dot(data[i]*(self.classes[fst]-self.classes[snd]),(masks[fst][snd] - masks[snd][fst]))
                    #if score < 0:
                    #    guess = snd
                elif mask_d == 1:
                    the_mask = masks[fst]
                    fst_sc = np.dot((data[i] * the_mask), self.classes[fst])
                    snd_sc = np.dot((data[i] * the_mask), self.classes[snd])
                    ratio = fst_sc / snd_sc
                    if 0 > ratio or ratio > threshold:
                        guess = snd
                else:
                    print("Error in determining mask dimensions")
                    return

                # Handle matrices
                matrix[fst][snd] += 1
                if guess == answer:
                    net_matrix[fst][snd] += 1
                else:
                    net_matrix[fst][snd] -= 1


            if answer == guess:
                correct += 1
                if guess == snd:
                    correct_r += 1
                    if debug:
                        print("Correctly selected the second:")
                        print(fst, snd)
                        #print(the_mask)
                        print(ratio)

            if fst == answer:
                correct_1 += 1
                if guess != fst:
                    wrong_1 += 1
                    if debug:
                        print("Wrongly selected the second:")
                        print(fst, snd)
                        #print(the_mask)
                        print(ratio)
            elif snd == answer:
                correct_2 += 1
                if guess != snd:
                    wrong_2 += 1
                    if debug:
                        print("Wrongly selected the first:")
                        print(fst, snd)
                        print(ratio)
            else:
                wrong_3 += 1
            count += 1
        return correct, correct_1, correct_2, correct_r, wrong_1, wrong_2, wrong_3, count, matrix, net_matrix

    def test_decider(self, data, label, threshold = 1, dominance = None, mask_t = "o", mask_d = 2, kernel_t = Config.config["kernel"], debug = None):
        assert self.D == data.shape[1]
        if debug is None:
            debug = self.debug

        # Make copy of inference model and data (for quantization)
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        #data_cp = self.preprocess_data(data, kernel_t)

        # fit
        correct = 0
        correct_1 = 0
        correct_2 = 0
        correct_r = 0

        wrong_1 = 0
        wrong_2 = 0
        wrong_3 = 0 # When the answer is not in top 2

        matrix = np.zeros((self.nClasses, self.nClasses))
        net_matrix = np.zeros((self.nClasses,self.nClasses))
        count = 0

        masks = self.mask_selector(mask_t, mask_d)
        if masks is None:
            print("OOF NO MASK OF THE TYPE FOUND")
            return None

        for i in range(data.shape[0]):
            answer = label[i]
            vals = batch_kernel(og_classes, data[i], kernel_t)
            order = list(reversed(np.argsort(vals)))
            fst = order[0]
            snd = order[1]
            guess = fst
            the_mask = None
            fst_sc = vals[fst]
            snd_sc = vals[snd]
            ratio = fst_sc/snd_sc
            if dominance is not None and ratio >= dominance:
                guess = fst
            elif not np.isnan(np.sum(masks[fst][snd])) and not np.isnan(np.sum(masks[snd][fst])):
                if mask_d == 2:
                    the_mask = masks[fst][snd] - masks[snd][fst]
                    #if np.sum(the_mask) == 0:
                    #    print("Found untrained mask (%d, %d) during testing" % (fst, snd))
                    #    print("Their submasks starts with:")
                    #    print(masks[fst][snd][0:10], masks[snd][fst][0:10])
                    #    print("Selecting fst by default")
                    #    continue
                    fst_sc = np.dot((data[i] * the_mask), self.classes[fst])
                    snd_sc = -np.dot((data[i] * the_mask), self.classes[snd])
                    diff = (fst_sc - snd_sc) * self.decider[fst][snd]
                    if diff < 0: # Also include cases where decider == 0
                        guess = snd
                    #score = np.dot(data[i]*(self.classes[fst]-self.classes[snd]),(masks[fst][snd] - masks[snd][fst]))
                    #if score < 0:
                    #    guess = snd
                elif mask_d == 1:
                    the_mask = masks[fst]
                    fst_sc = np.dot((data[i] * the_mask), self.classes[fst])
                    snd_sc = np.dot((data[i] * the_mask), self.classes[snd])
                    ratio = fst_sc / snd_sc
                    if 0 > ratio or ratio > threshold:
                        guess = snd
                else:
                    print("Error in determining mask dimensions")
                    return

                # Handle matrices
                matrix[fst][snd] += 1
                if guess == answer:
                    net_matrix[fst][snd] += 1
                else:
                    net_matrix[fst][snd] -= 1


            if answer == guess:
                correct += 1
                if guess == snd:
                    correct_r += 1
                    if debug:
                        print("Correctly selected the second:")
                        print(fst, snd)
                        #print(the_mask)
                        print(ratio)

            if fst == answer:
                correct_1 += 1
                if guess != fst:
                    wrong_1 += 1
                    if debug:
                        print("Wrongly selected the second:")
                        print(fst, snd)
                        #print(the_mask)
                        print(ratio)
            elif snd == answer:
                correct_2 += 1
                if guess != snd:
                    wrong_2 += 1
                    if debug:
                        print("Wrongly selected the first:")
                        print(fst, snd)
                        print(ratio)
            else:
                wrong_3 += 1
            count += 1
        return correct, correct_1, correct_2, correct_r, wrong_1, wrong_2, wrong_3, count, matrix, net_matrix



    def analyze_topn(self, data, label, dominance = None, mask_t="o", mask_d = 2, kernel_t = Config.config["kernel"]):
        assert self.D == data.shape[1]

        print("Analyzing score with mask type", mask_t, mask_d)

        # Make copy of inference model and data (for quantization)
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        # data_cp = self.preprocess_data(data, kernel_t)

        masks = None
        if mask_t == "o":
            if mask_d == 1:
                masks = self.mask1d
            elif mask_d == 2:
                masks = self.mask2d
        elif mask_t == "b":
            if mask_d == 1:
                masks = self.bmask1d
            elif mask_d == 2:
                masks = self.bmask2d
        if masks is None:
            print("WTF no mask type found")

        fst_scs = []
        snd_scs = []

        for i in range(data.shape[0]):
            answer = label[i]
            vals = batch_kernel(og_classes, data[i], kernel_t)
            order = list(reversed(np.argsort(vals)))
            fst = order[0]
            snd = order[1]
            the_mask = None
            fst_sc = vals[fst]
            snd_sc = vals[snd]
            ratio = fst_sc/snd_sc
            if answer in [fst,snd] and dominance is not None and ratio >= dominance:
                #print("Dominated")
                useless = 1
            elif answer in [fst, snd] and not np.isnan(np.sum(masks[fst][snd])) and not np.isnan(np.sum(masks[snd][fst])):
                if mask_d == 2:
                    the_mask = masks[fst][snd] - masks[snd][fst]
                else:
                    the_mask = masks[fst]
                if np.count_nonzero(the_mask) == 0:
                    print("Encounter unforseen mask:", fst, snd)
                    continue
                # Check 0
                #if np.sum(the_mask) == 0:
                #    print("Found untrained mask (%d, %d) of ratio %f"%( fst, snd, ratio))
                #    print("Their submasks starts with:")
                #    print(masks[fst][snd][0:10], masks[snd][fst][0:10])
                #    continue
                fst_sc = np.dot((data[i] * the_mask), self.classes[fst])
                snd_sc = -np.dot((data[i] * the_mask), self.classes[snd])
                if answer == fst:
                    fst_scs.append((fst_sc, snd_sc, i, fst, snd))
                else:
                    snd_scs.append((fst_sc, snd_sc, i, fst, snd))

        return np.asarray(fst_scs), np.asarray(snd_scs)

    def analyze(self, data, label, dominance = None, kernel_t=Config.config["kernel"]):
        assert self.D == data.shape[1]

        # Make copy of inference model and data (for quantization)
        og_classes = self.preprocess_classes(self.classes, kernel_t)
        # data_cp = self.preprocess_data(data, kernel_t)

        fst_scs = []
        snd_scs = []

        for i in range(data.shape[0]):
            answer = label[i]
            vals = batch_kernel(og_classes, data[i], kernel_t)
            order = list(reversed(np.argsort(vals)))
            fst = order[0]
            snd = order[1]
            fst_sc = vals[fst]
            snd_sc = vals[snd]
            if dominance is not None and dominance < fst_sc/snd_sc:
                continue
            if answer == fst:
                fst_scs.append((fst_sc, snd_sc, i, fst, snd))
            else:
                snd_scs.append((fst_sc, snd_sc, i, fst, snd))
        return np.asarray(fst_scs), np.asarray(snd_scs)



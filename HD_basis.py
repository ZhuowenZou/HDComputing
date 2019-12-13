import Config
import sys
import time
import numpy as np
from Config import config
import joblib
from enum import Enum

# enum for random vector generator type
class Genetator(Enum):
    Vanilla = 1
    Baklava = 2

# Generate one random vector of desired length and generation type
def generate_vector(vector_length, vector_type, param):
    if vector_type == "Gaussian":
        mu = param["mu"]
        sigma = param["sigma"]
        return np.random.normal(mu, sigma, vector_length)
    else:
        raise Exception("Vector type %s not recognized. Abort.\n" % vector_type)

class HD_basis:

    # required parameters for generator types from the dataset (not Config)
    param_req = {
        Genetator.Vanilla: ["nFeatures"],
        Genetator.Baklava: ["nFeatures"]
    }
    # general parameters from Config
    param_config = ["nFeatures", "D", "sparse", "s", "vector", "mu", "sigma", "binarize"]

    # gen_type: type of random vector generator
    # param: dictionary containing parameter of the generator
    def __init__(self, gen_type, param):
        # sanity check
        for req in self.param_req[gen_type]:
            if req not in param:
                sys.stderr.write("required parameters not received in HD_Basis, abort.\n")
                return
        self.param = dict()

        # scrape parameters from param then config
        for term in self.param_config:
            if term in param:
                self.param[term] = param[term]
            else:
                self.param[term] = config[term]

        start = time.time()
        self.gen_type = gen_type
        if gen_type == Genetator.Vanilla:
            self.vanilla(param)
        elif gen_type == Genetator.Baklava:
            self.baklava(param)
        end = time.time()
        print('Encoding time: ' + str(end - start))
        joblib.dump(self.basis, open("base.pkl", "wb"), compress=True)

    def vanilla(self, param):

        self.basis = []
        for i in range(param["D"]):
            self.basis.append(generate_vector(param["nFeatures"], param["vector"], param))
        self.basis = np.asarray(self.basis)
        sys.stderr.write("Generated vanilla HD basis of shape "+str(self.basis.shape))

    def baklava(self, param):
        pass

    def getBasis(self):
        return self.basis

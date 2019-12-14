import Config
import sys
import time
import math
import numpy as np
from Config import config
import joblib
from enum import Enum

from tqdm import tqdm_notebook

# enum for random vector generator type
class Generator(Enum):
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

# dump basis and its param into a file and return the name
def saveBasis(basis, param = None):
    if param is None:
        param = {"id": ""}
    filename = "base_%s.pkl" % param["id"]
    sys.stderr.write("Dumping basis into file: %s \n"%filename)
    joblib.dump((basis, param), open(filename, "wb"), compress=True)
    return filename

# Load basis from a file
def loadBasis(filename = "base_.pkl"):
    basis, param = joblib.load(filename)
    return basis, param

class HD_basis:

    # required parameters for generator types from the dataset (not Config)
    param_req = {
        Generator.Vanilla: ["nFeatures"],
        Generator.Baklava: ["nFeatures"]
    }
    # general parameters from Config
    param_config = ["nFeatures", "D", "sparse", "s", "vector", "mu", "sigma", "binarize"]

    # gen_type: type of random vector generator
    # param: dictionary containing parameter of the generator
    def __init__(self, gen_type, param):
        # sanity check
        for req in self.param_req[gen_type]:
            if req not in param:
                raise Exception("required parameters not received in HD_Basis, abort.\n")

        # Timestamp for uniquely identify a basis
        self.param = {"id": str(int(time.time()) % 10000)}

        # scrape parameters from param then config
        for term in self.param_config:
            if term in param:
                self.param[term] = param[term]
            else:
                self.param[term] = config[term]

        start = time.time()
        self.param["gen_type"] = gen_type
        if gen_type == Generator.Vanilla:
            self.vanilla()
        elif gen_type == Generator.Baklava:
            self.baklava()
        end = time.time()
        sys.stderr.write('Encoding time: %s \n' % str(end - start))
        self.filename = saveBasis(self.basis, self.param)

    def vanilla(self):

        sys.stderr.write("Generated vanilla HD basis of shape... ")
        self.basis = []
        #for i in range(param["D"]):
        for _ in tqdm_notebook(range(self.param["D"]), desc='vectors'):
            self.basis.append(generate_vector(self.param["nFeatures"], self.param["vector"], self.param))
        self.basis = np.asarray(self.basis)
        sys.stderr.write(str(self.basis.shape)+"\n")

    def baklava(self):
        pass

    def getBasis(self):
        return self.basis

    def getParam(self):
        return self.param

import Config
import sys
import time
import math
import numpy as np
from Config import config
import joblib
from enum import Enum

from tqdm import tqdm_notebook

#encode one vector/sample into a HD vector
def encodeDatum(datum, basis, noises):
    size = basis.shape[0]
    assert size == noises.shape[0]
    data = datum
    encoded = np.empty(size)
    for i in range(size):
        encoded[i] = np.cos(np.dot(datum, basis[i]) + noises[i]) * np.sin(np.dot(datum, basis[i]))
    return encoded

# dump basis and its param into a file, return the name of file
def saveEncoded(encoded, id = "", data_type = "unknown"):
    filename = "encoded_%s_%s.pkl" % (id, data_type)
    sys.stderr.write("Dumping data into %s \n"%filename)
    joblib.dump(encoded, open(filename, "wb"), compress=True)
    return filename

# Load basis from a file
def loadEncoded(filename):
    encoded= joblib.load(filename)
    return encoded

# encode data using the given basis
# noise: default Gaussian noise
def encodeData(data, basis, noise = True):
    start = time.time()
    sys.stderr.write("Encoding data of shape %s\n"%str(data.shape))
    assert data.shape[1] == basis.shape[1]
    noises = []
    encoded = []
    if noise:
        noises = np.random.uniform(0, 2 * math.pi, basis.shape[0])
    else:
        noises = np.zeros(basis.shape[0])
    for i in tqdm_notebook(range(len(data)), desc='samples encoded'):
        encoded.append(encodeDatum(data[i], basis, noises))
    end = time.time()
    sys.stderr.write("Time spent: %d sec\n" % int(end - start))
    return np.asarray(encoded)



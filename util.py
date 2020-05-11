import joblib
import numpy as np

def dump_log(param, train_acc, test_acc, filename):
    joblib.dump((param, train_acc, test_acc), open(filename+".pkl", "wb"), compress=True)
    file = open(filename+".txt", "a")
    msg = str(100*max(train_acc)) + " " + str(100*max(test_acc)) + " " +\
        str(len(train_acc)) + " " + str(np.argmax(test_acc) + 1) + "\n"
    file.write(msg)
    file.close()
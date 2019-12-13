import Config
import joblib
import sys
import random
import math
import numpy as np
import parse_example

D = Config.D
sparse = Config.sparse

def kitchen(datum,base,base2):
  data = np.array([])
  size = int(D)
  #print(size)
  #print(base2.shape[0])
  #sys.exit(0)
  #if expand < 1:
    #data = datum
  #else:
    #for i in range(expand + 1):
      #data = np.append(data,datum)
  data = datum
  encoded = np.empty(size)
  assert size == base.shape[0]
  assert size == base2.shape[0]
  for i in range(size):
     #encoded[i] = np.cos(np.dot(datum,base2[i]))
     encoded[i] = np.cos(np.dot(datum,base2[i]) + base[i])*np.sin(np.dot(datum,base2[i]))
     #encoded[i] = encoded[i] / math.sqrt(size)
      
     #encoded[i] = np.exp(-1*base[i]*np.dot(datum,base2[i]))#np.cos(data[i] + base[i])
     #encoded[i] = np.dot(datum,base2[i])
  #print(encoded.shape)
  #sys.exit(0)
  return encoded

def encode(data,base):
  base2 = joblib.load("base.pkl")
  newData = list()
  for i in range(data.shape[0]):
    newData.append(kitchen(data[i],base,base2))
  return np.asarray(newData)


def condense(data,labels):
    nClasses = np.unique(labels)
    nClasses = len(nClasses)
    data = np.asarray(data)
    num = data.shape[0]
    dim = data.shape[1]
    smaller = np.zeros((nClasses,dim))
    for i in range(0,num):
        smaller[labels[i]] = smaller[labels[i]] + data[i]
    return smaller,np.arange(nClasses)

"""
def retrain (model,traindata,trainlabels,retNum,rate):
    # we assume one model per class, i.e. label 0 is model 0 is the 0th index
    # of the model, etc.
    modelLabels = np.arange(len(model))
    # retrain iterations or epochs 
    for ret in range(retNum):
        # go stochastically, in random order
        r = list(range(len(traindata)))
        random.shuffle(r)
        correct = 0
        for i in r:
            query = traindata[i]
            answer = trainlabels[i]
            #guess = closestGuess(query,model,modelLabels)
            maxVal = -1
            for m in range(len(model)):
              val = kernel(model[m],query)
              if val > maxVal:
                maxVal = val
                guess = m
            if guess != answer:
                # if wrongly matched, use naive perceptron rule: 
                model[guess] = model[guess] - rate*query
                model[answer] = model[answer] + rate*query
            else:
                correct = correct + 1
        print('Retraining epoch: ' + str(ret) + ' Epoch accuracy:' + str(correct / len(traindata)))
    return model
"""


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



def binarizeSamples(a,b,data,labels):
  newData = list()
  newLabels = list()
  for i in range(data.shape[0]):
    sample = data[i]
    answer = labels[i]
    if answer == a:
      newData.append(sample)
      newLabels.append(1)
    elif answer == b:
      newData.append(sample)
      newLabels.append(-1)
  return np.asarray(newData), np.asarray(newLabels)

def binarize(datum,big,small):
  return np.where(datum > 0,big,small)

def binarizeAll(data,big,small):
  for i in range(data.shape[0]):
    data[i] = binarize(data[i],big,small)
  return data

#def normalize(data):


def load(directory,dataset):
    traindirectory = directory
    traindataset = dataset
    testdirectory = directory
    testdataset = dataset

    pathTrain = '../dataset/'
    pathTrain = pathTrain + traindirectory
    pathTrain = pathTrain + '/' + traindataset + '_train.choir_dat'

    pathTest = '../dataset/'
    pathTest = pathTest + testdirectory
    pathTest = pathTest + '/' + testdataset + '_test.choir_dat'

    print('Loading datasets')
    nTestFeatures, nTestClasses, testdata, testlabels = parse_example.readChoirDat(pathTest)
    nTrainFeatures, nTrainClasses, traindata, trainlabels = parse_example.readChoirDat(pathTrain)
    traindata = np.asarray(traindata)
    trainlabels = np.asarray(trainlabels)
    testdata = np.asarray(testdata)
    testlabels = np.asarray(testlabels)
    return traindata, trainlabels, testdata, testlabels,nTrainFeatures,nTrainClasses

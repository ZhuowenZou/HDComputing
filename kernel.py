import Config
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import time
import sys
import createNormalBase
import math
import numpy as np
import random
import joblib
import parse_example
import KernelFunctions
from enum import Enum

class Update_T(Enum):
  FULL = 1
  PARTIAL = 2
  RPARTIAL = 3


sgn = KernelFunctions.sgn


def update(weights, weight, mask, guess, answer, rate, update_type):
  sample = weight * mask
  if update_type == Update_T.FULL:
    weights[guess] = weights[guess] - rate * sample
    weights[answer] = weights[answer] + rate * sample
  elif update_type == Update_T.PARTIAL:
    weights[guess] = weights[guess] - rate * sample
    weights[answer] = weights[answer] + rate * weight
  elif update_type == Update_T.RPARTIAL:
    weights[guess] = weights[guess] - rate * weight
    weights[answer] = weights[answer] + rate * sample

def trainMulticlass(iterations, rate, update_type):

  weights = np.zeros((nTrainClasses,traindata.shape[1]))
  correct = 0
  t = 0
  accuracies = []
  while (correct / traindata.shape[0]) != 1 and t < iterations:

    # Mask for dropout
    dim = traindata.shape[1]
    mask = np.asarray([1 for i in range(dim)])
    for i in np.random.choice(dim, int(dim*(Config.dropout_rate)), replace=False):
      mask[i] = 0

    r = list(range(traindata.shape[0]))
    random.shuffle(r)
    correct = 0
    count = 0
    for i in r:
      sample = traindata[i] * mask
      assert traindata[i].shape == mask.shape

      answer = trainlabels[i]
      maxVal = -1
      guess = -1
      for m in range(nTrainClasses):
        val = kernel(weights[m],sample)
        if val > maxVal:
          maxVal = val
          guess = m 
      if guess != answer:
        update(weights, traindata[i], mask, guess, answer, rate, update_type)
      else:
        correct += 1
      count += 1
    accuracy = 100*testMulticlass(weights)
    accuracies.append(accuracy)
    print("Iteration: ",t,"Train Accuracy: ",correct / count,"Test Accuracy: ", accuracy)
    t += 1
  print('Max Accuracy: ' + str(max(accuracies)))

  accuracies = np.asarray(accuracies)
  s ="Max test Accuracy " + str(max(accuracies))+ " reached at iteration " + str(np.argmax(accuracies))+ ", with current iteration " + str(t-1) + "\n"
  log = open("./logfile/" + Config.dataset + "_" + str(Config.D) + "_" + str(Config.train_percent) + "_" + str(Config.dropout_rate) + "_" + str(update_type.value)+"_max.txt", "a")
  log.write(s)

def trainMulticlassBinary(iterations,rate):
  weights = np.zeros((nTrainClasses,traindata.shape[1]))
  binaryWeights = np.copy(weights)
  correct = 0
  t = 0
  while (correct / traindata.shape[0]) != 1:
    r = list(range(traindata.shape[0]))
    random.shuffle(r)
    correct = 0
    count = 0
    for i in r:
      sample = traindata[i]
      answer = trainlabels[i]
      maxVal = -1
      guess = -1
      for m in range(nTrainClasses):
        val = kernel(binaryWeights[m],sample)
        if val > maxVal:
          maxVal = val
          guess = m 
      if guess != answer:
        weights[guess] = weights[guess] - rate*sample
        weights[answer] = weights[answer] + rate*sample
        binaryWeights = np.copy(weights)
        binaryWeights = KernelFunctions.binarizeAll(binaryWeights, 1, -1)
      else:
        correct += 1
      count += 1
    print("Iteration: ",t,"Train Accuracy: ",correct / count,"Test Accuracy: ",100*testMulticlass(binaryWeights))
    t += 1


def testMulticlass(weights):
  correct = 0
  guess = 0
  for i in range(testdata.shape[0]):
    sample = testdata[i]
    answer = testlabels[i]
    maxVal = -1
    for m in range(nTrainClasses):
      val = kernel(weights[m],sample)
      if val > maxVal:
        maxVal = val
        guess = m
    if guess == answer:
      correct += 1
  return correct / testdata.shape[0]


# take in the whole train dataset, randomly pick
# percent of the dataset to return
# !! randomness is within label !!
def one_shot( data, labels, percent):

  #print("labels: ", labels)

  #categorize all data
  sorted_data = dict()
  for i in range(len(data)):
    if labels[i] in sorted_data:
      sorted_data[labels[i]].append(data[i])
    else:
      sorted_data[labels[i]] = [data[i]]

  train_data = []
  train_labels = []

  for label in sorted_data.keys():

    indeces = [i for i in range(len(sorted_data[label]))]
    indeces = np.asarray(indeces)
    np.random.shuffle(indeces)
    #print("shuffles indeces:", indeces)
    for i in range(min(len(sorted_data[label]), max(1, int(len(sorted_data[label])*percent)))):
      train_data.append(sorted_data[label][i])
      train_labels.append(label)

  #perm = np.random.permutation(len(train_data))
  return np.asarray(train_data), np.asarray(train_labels)



# update train percent if applicable
if len(sys.argv) >= 3:
  Config.train_percent = float(sys.argv[2])
  Config.dropout_rate = float(sys.argv[3])

run_iteration = 1

print("Traindata percentage:", Config.train_percent)
print("Drop rate:", Config.dropout_rate)
print("Run: ", run_iteration)

directory = Config.directory 
dataset = Config.dataset
kernel = KernelFunctions.kernel
init = int(sys.argv[1])

#fulldata vs traindata: all the data and the data to train on for one shot
fulldata, fulllabels, testdata, testlabels,nTrainFeatures, nTrainClasses = KernelFunctions.load(directory,dataset)
traindata, trainlabels = one_shot(fulldata, fulllabels, Config.train_percent)


if init == 1:
  D = KernelFunctions.D

  #Normalizing train data
  traindata = sklearn.preprocessing.normalize(traindata,norm='l2')
  testdata = sklearn.preprocessing.normalize(testdata,norm='l2') 

  mu = Config.mu
  sigma = Config.sigma #/ 20#1 / (math.sqrt(617)) #/ 24#1 #/ (1.4)
  if Config.sparse == 1:
    createNormalBase.createSparse(D, nTrainFeatures, mu, sigma, Config.s)
  else:
    createNormalBase.create(D,nTrainFeatures,mu,sigma)
  size = int(D)
  base = np.random.uniform(0,2*math.pi,size)
  start = time.time()

  # Dump the selected train_data somewhere so that the MLP can use it
  joblib.dump(traindata,open('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) +"_" +\
                             str(Config.D) + "_" + str(Config.train_percent) + "_" + str(run_iteration)+ \
                             '_selected_train.pkl',"wb"),compress=True)
  joblib.dump(trainlabels,open('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) +"_" + \
                             str(Config.D) + "_" + str(Config.train_percent) + "_" + str(run_iteration)+ \
                             '_selected_labels.pkl',"wb"),compress=True)

  # HD part
  traindata = KernelFunctions.encode(traindata,base)

  assert traindata.shape[0] == trainlabels.shape[0]
  print("Encoding training time",time.time() - start)
  start = time.time()
  testdata = KernelFunctions.encode(testdata,base)
  print('Encoding testing time',time.time() - start)
  if Config.sparse == 1:
    joblib.dump(traindata,open('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'train.pkl',"wb"),compress=True)
    joblib.dump(testdata,open('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'test.pkl','wb'),compress=True)
  else:
    joblib.dump(traindata,open('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + 'train.pkl',"wb"),compress=True)
    joblib.dump(testdata,open('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + 'test.pkl','wb'),compress=True)
  print(traindata.shape,trainlabels.shape)
  print(testdata.shape,testlabels.shape)
  sys.exit(0)
else:
  if Config.sparse == 1:
    traindata = joblib.load('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'train.pkl')
    testdata = joblib.load('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'test.pkl')
  else:
    traindata = joblib.load('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + 'train.pkl')
    testdata = joblib.load('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + 'test.pkl')

  if Config.binarize == 1:
    traindata = KernelFunctions.binarizeAll(traindata, 1, -1)
    testdata = KernelFunctions.binarizeAll(testdata, 1, -1)
  pass

for i in range(Config.iter_per_encoding):
  if Config.binaryModel == 1:
    trainMulticlassBinary(300, Config.rate)
  else:
    #trainMulticlass(300, Config.rate, Update_T.FULL)
    #trainMulticlass(300, Config.rate, Update_T.PARTIAL)
    trainMulticlass(300, Config.rate, Update_T.RPARTIAL)




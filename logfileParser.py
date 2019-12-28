from os import listdir
from os.path import isfile, join
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def average_time(s):
	nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)

	D = cp = nb = 0
	# Handle ISOMAP files
	if s[0] == "i":
		D, cp, nb = float(nums[0]), float(nums[1]), float(nums[2])
	elif s[0] == "m":
		D, cp, nb = float(nums[0]), float(nums[1]), float(nums[2])

	file = open(mypath+"/"+s, "r")
	train_accs = 0
	test_accs = []
	iterations = []
	prep = trans = -1
	s = file.readline()
	while s != "":
		num = re.findall(r"[-+]?\d*\.\d+|\d+", s)
		if len(num) >= 2:
			prep, trans = num[0], num[1]
		s = file.readline()
	file.close()
	return D, cp, nb, float(prep), float(trans)

def average(s):
	nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)

	D = cp = nb = 0
	# Handle baseline:
	if s[0] == "b":
		D = float(nums[0])
	# Handle ISOMAP files
	elif s[0] == "i":
		D, cp, nb = float(nums[0]), float(nums[1]), float(nums[2])
	elif s[0] == "m":
		D, cp, nb = float(nums[0]), float(nums[1]), float(nums[2])

	file = open(mypath+"/"+s, "r")
	train_accs = []
	test_accs = []
	iterations = []

	s = file.readline()
	while s != "":
		num = re.findall(r"[-+]?\d*\.\d+|\d+", s)
		if len(num) >= 4:
			train_acc, test_acc, iteration = num[0], num[1], num[3]
			train_accs.append(train_acc)
			test_accs.append(test_acc)
			iterations.append(iteration)
		s = file.readline()
	file.close()
	return D, cp, nb, np.min(np.asarray(train_accs).astype(np.float)), np.mean(np.asarray(test_accs).astype(np.float)), np.mean(np.asarray(iterations).astype(np.float))

# Collect results from HD
mypath = "./logfile"
filenames = [f for f in listdir(mypath+"/") if isfile(join(mypath+"/", f))]

for manitype in ["baseline", "isomap", "modified"]:
	max_train = dict()
	avg_accu = dict()
	avg_iter = dict()
	avg_prep = dict()
	avg_trans = dict()

	for s in filenames:
		if s[0] != manitype[0]:
			continue
		if s[-4:] == ".txt":
			# handle time
			if s[-5] == "e":
				print(s, " is a time txt.")
				D, cp, nb, prep, trans = average_time(s)
				if D not in avg_prep:
					avg_prep[D] = dict()
					avg_trans[D] = dict()
				if cp not in avg_prep[D]:
					avg_prep[D][cp] = dict()
					avg_trans[D][cp] = dict()
				avg_prep[D][cp][nb] = prep
				avg_trans[D][cp][nb] = trans
			else:
				print(s, " is a result txt.")
				D, cp, nb, train, test, iter = average(s)
				if D not in avg_accu:
					max_train[D] = dict()
					avg_accu[D] = dict()
					avg_iter[D] = dict()
				if cp not in avg_accu[D]:
					max_train[D][cp] = dict()
					avg_accu[D][cp] = dict()
					avg_iter[D][cp] = dict()
				if nb not in avg_accu[D][cp]:
					max_train[D][cp][nb] = train
					avg_accu[D][cp][nb] = [test]
					avg_iter[D][cp][nb] = [iter]
				else:
					max_train[D][cp][nb] = max(train, max_train[D][cp][nb])
					avg_accu[D][cp][nb].append(test)
					avg_iter[D][cp][nb].append(iter)
	print(avg_iter)
	print(avg_prep)
	dumper = open(manitype+"_LogSum.txt", "w")
	dumper.write("The max train accuracy, avg test accuracy, avg iteration, prep time, and transform time:\n")
	for D in max_train:
		dumper.write("Dimention %d: \n"%D)
		for cp in max_train[D]:
			dumper.write("\t Component %d: \n"%cp)
			for nb in max_train[D][cp]:
				sys.stderr.write("Dumping: %d %d %d\n"%(D, cp, nb))
				if manitype!="baseline":
					dumper.write("\t \t Neighbor %d: %.3f \t %.3f \t %.3f \t %.3f \t %.3f \n"\
							 %(nb, float(np.mean(max_train[D][cp][nb])), float(np.mean(avg_accu[D][cp][nb])),\
							   float(np.mean(avg_iter[D][cp][nb])), avg_prep[D][cp][nb], avg_trans[D][cp][nb]))
				else:
					dumper.write("\t \t Neighbor %d: %.3f \t %.3f \t %.3f \n" \
								 %(nb, float(np.mean(max_train[D][cp][nb])), float(np.mean(avg_accu[D][cp][nb])), \
								   float(np.mean(avg_iter[D][cp][nb]))))


pass
"""
dumper = open("LogSum.txt", "w")
msg ="Data rate: "+ str(rate) + ", Drop rate: " + str(drop) + ", accuracy: " + str(np.mean(accu)) + " (" + str(np.std(accu)) + ") iterations: " +\
	str(np.mean(iter)) +" (" + str(np.std(accu)) + ")\n"
dumper.write(msg)
dumper.close()

for key_rate in table:
	print(key_rate, end = " & ")
	for key_drop in table[key_rate]:
		print(str(table[key_rate][key_drop][0]) + "(" + str(table[key_rate][key_drop][1]) + ")", end=  " & ")
	print()


for key_rate in table:
	print(key_rate, end = " & ")
	for key_drop in table[key_rate]:
		print(key_drop, end=  " & ")
	print()
	break
"""

"""
mypath = "ExtraSensory_MLP/"
filenames = [f for f in listdir(mypath+"/") if isfile(join(mypath+"/", f))]
dumper = open(mypath+"_LogSum.txt", "w")
rates_MLP = []
avg_accu_MLP = []
std_accu_MLP = []
avg_iter_MLP = []
std_iter_MLP = []
for s in filenames:
	rate, accu, iter = average(s)
	accu *= 100
	# omit previous logSum
	if len(accu) != 0:
		rates_MLP.append(rate)
		avg_accu_MLP.append(np.mean(accu))
		std_accu_MLP.append(np.std(accu))
		avg_iter_MLP.append(np.mean(iter))
		std_iter_MLP.append(np.std(iter))
		msg ="Rate: "+ str(rate) + " accuracy: "+ str(np.mean(accu))+ " (" + str(np.std(accu)) + ") iterations: " +\
			str(np.mean(iter)) +" (" + str(np.std(accu)) + ")\n"
		dumper.write(msg)
print(rates_MLP)
dumper.close()


#Plot_graph

mypath = "ExtraSensory"
plt.errorbar(rates, avg_accu, yerr = std_accu, label = "HD")
plt.errorbar(rates_MLP, avg_accu_MLP, yerr = std_accu_MLP, label="MLP")

#plt.fill_between(rates, avg_accu - std_accu, avg_accu + std_accu, alpha = 0.4)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.suptitle("Average accuracies of MLP and HD models for " +mypath, fontsize=12)
plt.ylabel('Accuracies')
plt.xlabel('Data percentage')
plt.show()
"""
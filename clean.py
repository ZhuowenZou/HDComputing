# Clean up generated dump files, including:
#   selected training set for MLP
#   logfiles
import Config
import os
from os import listdir
from os.path import isfile, join

# Usage: set in Config the name of the data, then run this. Delete everything mentioned above associated with the data

#Prompt
print("deleting everything persistent w.r.t dataset:", Config.directory)
confirm = ""
while confirm not in ["y", "n"]:
	confirm = input("OK to push to continue [Y/N]? ").lower()
if confirm == 'n':
	print("K...")
	exit(0)

# start purging

#purge log file
path_to_logfile = "./logfile/"+Config.directory
filenames = [f for f in listdir(path_to_logfile+"/") if isfile(join(path_to_logfile+"/", f))]
for filename in filenames:
	print("Purged: ", filename)
	os.remove(path_to_logfile + "/" + filename)

#purge log file
path_to_datapkl = "../dataset/"+Config.directory
filenames = [f for f in listdir(path_to_datapkl+"/") if isfile(join(path_to_datapkl+"/", f))]
for filename in filenames:
	#print(filename[-19:])
	if filename[-19:] == "_selected_train.pkl":
		print("Purged: ", filename)
		os.remove(path_to_datapkl + "/" + filename)
	elif filename[-20:] == "_selected_labels.pkl":
		print("Purged: ", filename)
		os.remove(path_to_datapkl + "/" + filename)


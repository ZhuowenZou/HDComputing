import Config
import os

for rate in Config.data_percentages:
	print("Train:")
	os.system("python kernel.py 1 " + str(rate) + " 0")
	for drop in Config.drop_percentages:
		print("Test:")
		os.system("python kernel.py 0 " + str(rate) + " " + str(drop))






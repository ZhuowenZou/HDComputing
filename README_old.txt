The directory contains:
	logfile:		
		[DATASET]		directories
		[DATASET]_MLP		directories
		dataInfo		directories
		logfileParser.py
		Graph			directories
	src:
		[source codes]
	README.txt	
	Project Summary(Simplified)-MI.docx

Where 
[DATASET] denotes the name of the dataset.
[source code] are source codes for the testing.

1. For each directory [DATASET], there exist:
	_LogSum
	[DATASET]_[D]_[Percent]_max.txt
Where  
[D] denotes the dimension of the hyper-vectors, and
[Percent] denotes the percentage of data used.
_Logsum gives the average and std of the accurawcy for each percentage.

[DATASET]_MLP is the same, but for MLP.

2. The dataInfo directories contains the # of classes, # of features/datum: 608, and 
# of members for each classes (as an array) for each dataset.

3. logfileParse.py: a script for generating _LogSum.txt for a specific dataset and plot
the graph for accuracy for both HD and MLP.
Usage: 
	1. Change the variable mypath in the file to the name of the directory you want.
	2. Run the script. The script will (only) automatically ignore the file 
	DataInfo_[DATASET].txt under the directory, parse everything else, calc the
	average of the first ten trials in each file, and write the results in 
	./[DATASET]_Logsum.txt.
Example graphs are under the directory Graph.

4. Additional source file:
	automation: automatically train all the HD models. Look at the code for more detail.
	MLP: automatically train all the MLP models.
	clean: delete every log and dumped data generated for a specific dataset

5. Project Summary: the report and charts aggregating selected results from the raw files.

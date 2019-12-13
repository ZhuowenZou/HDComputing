import Config
import joblib
import KernelFunctions
import json
from sklearn import neural_network as nn
from sklearn import model_selection as ms

sgn = KernelFunctions.sgn


def train_MLP(run_iteration, percentage):

	kernel = KernelFunctions.kernel
	fulldata, fulllabels, testdata, testlabels, nTrainFeatures, nTrainClasses = KernelFunctions.load(directory, dataset)

	# Load the selected train_data somewhere so that the MLP can use it
	traindata = joblib.load(	'../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + "_" + \
                            str(Config.D) + "_" + str(percentage) + "_" + str(run_iteration) + \
                            '_selected_train.pkl')
	trainlabels = joblib.load(	'../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + "_" + \
	                            str(Config.D) + "_" + str(percentage) + "_" + str(run_iteration) + \
	                            '_selected_labels.pkl')

	estimator = nn.MLPClassifier((512, 512), learning_rate='constant',
	                             learning_rate_init=Config.rate, max_iter= 260, validation_fraction = 0.1)

	estimator.fit(traindata, trainlabels)
	score = estimator.score(testdata, testlabels)
	print("Default setting accuracy: ", score)

	param_grid = {
		'hidden_layer_sizes': [(512,512), (1024,1024), (256, 256, 256)],
		#'hidden_layer_sizes': [(512, 512), (1024, 1024)],
		#'validation_fraction': [0.05, 0.1, 0.2],
		'learning_rate_init': [0.02, Config.rate, 0.05, 0.1, 0.2]
		#'learning_rate_init': [0.02]
	}

	gridS = ms.GridSearchCV(estimator, param_grid, cv=3, verbose = 10)
	gridS.fit(traindata, trainlabels)
	best_estimator = gridS.best_estimator_
	best_score = best_estimator.score(testdata, testlabels)
	print(gridS.best_params_)
	print(gridS.best_score_)

	s = "Max test Accuracy reached with " + str(percentage) + " data: " + \
	    str(best_score) + ", iteration " + str(run_iteration) + " and parameter" + json.dumps(gridS.best_params_) + "\n"
	log = open("./logfile/" + Config.dataset + "_" + str(Config.D) + "_" + str(percentage) + "_max_MLP.txt", "a")
	log.write(s)
	print(s)
	log.close()


directory = Config.directory
dataset = Config.dataset
for rate in Config.data_percentages:
	for i in range(Config.iter_per_trial):
		print("Current rate and iteration: ", rate, i)
		train_MLP(i, rate)

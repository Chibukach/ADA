
import pandas as pd                                     
import numpy as np                                      
import os                                               
import matplotlib.pyplot as plt                         
import scipy.stats.mstats as ssm                        

from scipy.stats import gaussian_kde as kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import validation_curve
from config import *


#Create the model/estimator depending on whether it is:
#rfc - Random forest classifier
#k_means - K Means clustering

def create_mdl(cfg,modele_type = "rfc"):

	if modele_type == "rfc":
		estimator = RandomForestClassifier(n_estimators=cfg.n_estimators, criterion=cfg.criterion, 
			max_depth=cfg.max_depth, min_samples_split=cfg.min_samples_split, min_samples_leaf=cfg.min_samples_leaf, 
			min_weight_fraction_leaf=cfg.min_weight_fraction_leaf, max_features=cfg.max_features, max_leaf_nodes=cfg.max_leaf_nodes,
			bootstrap=cfg.bootstrap, oob_score=cfg.oob_score, n_jobs=cfg.n_jobs, 
			random_state=cfg.random_state, verbose=cfg.verbose, warm_start=cfg.warm_start, class_weight=cfg.class_weight)

	elif modele_type == "k_means":
		estimator = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
			precompute_distances='auto', verbose=0, random_state=None, 
			copy_x=True, n_jobs=1, algorithm='auto')

	return estimator


#compute the silhoutte score 
def kmeans_silhouette(k,data):
	cfg = config_kmeans(n_clusters=k)
	kmeans = create_mdl(cfg,modele_type="k_means")
	kmeans.fit(data)
	labels = kmeans.labels_
	score = silhouette_score(data, labels, metric='euclidean')
	return score

#compute the accuracy score
def kmeans_accuracy(k,data,labels):
	cfg = config_kmeans(n_clusters=k)
	kmeans = create_mdl(cfg,modele_type="k_means")
	kmeans.fit(data)
	score = accuracy_score(labels,kmeans.predict(data))
	return score

#train the models for random forest classifier and k_means
def train(X,Y,cfg,modele_type="rfc",cross_valid_bool=False,test_set_size=0.3,verbose=1,k_fold=None):

	estimator = create_mdl(cfg,modele_type=modele_type)

	if verbose == 1:
		print (estimator)

	if modele_type =="rfc":
     #Uses the two strategies of cross validation and train/split for rfc
		if cross_valid_bool==True:
			assert k_fold, 'you have to add a k_fold number if cross_valid_bool = True' 
			#kfold =  KFold(n_splits = k_fold)

			scores = cross_val_score(estimator, X, Y, cv=k_fold, scoring='accuracy')
			if verbose ==1:
				print ('The scores among the {}-Fold are {}'.format(k_fold, scores))
				print ('The accuracy of this model is {}'.format(scores.mean()))

			return scores.mean()

		else: ## train/testsplit
			X_train, X_test, Y_train, Y_test = train_test_split(
				X,Y,test_size = test_set_size,random_state=42)
			## random_state constant to be able to compare model 
			estimator.fit(X_train,Y_train)
			y_pred = estimator.predict(X_test)
   
             #retrieve the score metrics calculated from TP,TN,FP,FN values
			accuracy = accuracy_score(Y_test,y_pred)
			precision = precision_score(Y_test,y_pred)
			recall = recall_score(Y_test,y_pred)
			confusion_mat = confusion_matrix(Y_test,y_pred)
			TP = confusion_mat[1,1]
			TN = confusion_mat[0,0]
			FP = confusion_mat[0,1]
			FN = confusion_mat[1,0]

             #dataframe to display the confusion matrix  
			d = { "Predicted white (0)" : ["TN="+str(TN),"FN="+str(FN)],
                   "Predicted black (1)" : ["FP="+str(FP),"TP="+str(TP)]
                  }
			df = pd.DataFrame(data = d, index = ["Actually white (0)","Actually black (1)"])
                  

			if verbose ==1:
				  print ('The accuracy of this model is {}'.format(accuracy))
				  print ('The precision score of this model is {}'.format(precision))
				  print ('The recall score of this model is {}'.format(accuracy))
				  print('\n')
				  print(df)

                  
			return estimator

	elif modele_type =="k_means": ## kmeans model
		X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3,random_state=42)
		## random_state constant to be able to compare model 
		N,n_features = X.shape
		start=0
		silhouettes = []
		accuracy = []
		silhouettes.append(kmeans_silhouette(2,X_train))
		accuracy.append(kmeans_accuracy(2,X_test,Y_test))
		for i in range(0,n_features-1):
			X_train.drop(X_train.columns[0],axis=1,inplace=True)
			X_test.drop(X_test.columns[0],axis=1,inplace=True)
			silhouettes.append(kmeans_silhouette(2,X_train))
			accuracy.append(kmeans_accuracy(2,X_test,Y_test))
		# PLOT silhouettes
		plt.plot(range(n_features,0,-1), silhouettes, 'ro-', lw=2)
		plt.title('Silhouette coefficient plot')
		plt.xlabel('Number of features')
		plt.ylabel('Silhouette coefficient')
		plt.ylim(0,0.8)
		plt.xlim(1,n_features)
		plt.show()
		# PLOT accuracy
		plt.plot(range(n_features,0,-1), accuracy, 'ro-', lw=2)
		plt.title('Accuracy plot')
		plt.xlabel('Number of features')
		plt.ylabel('Accuracy')
		plt.ylim(0,0.8)
		plt.xlim(1,n_features)
		plt.show()

#standard graph plot implementation of the feature importances
#http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
def plot_forest_importances(rfc,X,verbose = 1):

	importances = rfc.feature_importances_
	std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
	indices = np.argsort(importances)[::-1]
	print("Feature ranking:")

	for f in range(X.shape[1]):
	    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], X.columns[indices[f]],importances[indices[f]]))

	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(X.shape[1]), importances[indices],
	       color="r", yerr=std[indices], align="center")
	plt.xticks(range(X.shape[1]), indices)
	plt.xlim([-1, X.shape[1]])
	plt.show()

def plot_validation_curve(estimator, X, y, param_name, param_range, cv=2):

	train_scores, valid_scores = validation_curve(estimator, X, y,param_name , param_range, 
                                              n_jobs=15,cv=cv)
	plt.figure()
	plt.plot(train_scores, color='b', label="Scores on training sets")
	plt.plot(valid_scores, color='r', label="Scores on test set")
	plt.title(param_name)
	plt.ylabel('score')
	plt.legend(loc='center right')
	plt.show()

#standard graph plot implementation of the learning curve
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
  
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

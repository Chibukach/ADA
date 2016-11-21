import pandas as pd                                     
import numpy as np                                      
import os                                               
import matplotlib.pyplot as plt                         
import scipy.stats.mstats as ssm                       
from scipy.stats import gaussian_kde as kde
from sklearn.ensemble import RandomForestClassifier

#Configuration class file for the random forest classifier
class config_rfc(object):
	"""docstring for Config_rfc"""
	def __init__(self, n_estimators=10,criterion='gini',max_features='auto',
		max_depth=None,min_samples_split=2,min_weight_fraction_leaf=0.0, min_samples_leaf=1,
		max_leaf_nodes=None,bootstrap=True,oob_score=False,n_jobs=1,
		random_state=None,verbose=0,warm_start=False,class_weight=None):
		
		self._n_estimators = n_estimators
		self._criterion = criterion
		self._max_features = max_features
		self._max_depth = max_depth
		self._min_samples_split = min_samples_split
		self._min_weight_fraction_leaf = min_weight_fraction_leaf
		self._min_samples_leaf = min_samples_leaf
		self._max_leaf_nodes = max_leaf_nodes
		self._bootstrap = bootstrap
		self._oob_score = oob_score
		self._n_jobs =  n_jobs
		self._random_state = random_state
		self._verbose =  verbose
		self._warm_start = warm_start
		self._class_weight = class_weight


	@property
	def n_estimators(self):
		return self._n_estimators
	@property
	def criterion(self):
		return self._criterion
	@property
	def max_features(self):
		return self._max_features
	@property
	def max_depth(self):
		return self._max_depth
	@property
	def min_samples_split(self):
		return self._min_samples_split
	@property
	def min_weight_fraction_leaf(self):
		return self._min_weight_fraction_leaf
	@property
	def min_samples_leaf(self):
		return self._min_samples_leaf
	@property
	def max_leaf_nodes(self):
		return self._max_leaf_nodes
	@property
	def bootstrap(self):
		return self._bootstrap
	@property
	def oob_score(self):
		return self._oob_score
	@property
	def n_jobs(self):
		return self._n_jobs
	@property
	def random_state(self):
		return self._random_state
	@property
	def verbose(self):
		return self._verbose
	@property
	def warm_start(self):
		return self._warm_start
	@property
	def class_weight(self):
		return self._class_weight

#Configuration class file for the kmeans clustering
class config_kmeans(object):
	"""docstring for Config_k_means"""
	def __init__(self,n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
			precompute_distances='auto', verbose=0, random_state=None, 
			copy_x=True, n_jobs=1, algorithm='auto'):
	
		self._n_clusters = n_clusters
		self._init = init
		self._n_init = n_init
		self._max_iter = max_iter
		self._tol = tol
		self._precompute_distances = precompute_distances
		self._verbose = verbose
		self._random_state = random_state
		self._copy_x = copy_x
		self._n_jobs = n_jobs
		self._algorithm = algorithm


	@property
	def n_clusters(self):
		return self._n_clusters
	@property
	def init(self):
		return self._init
	@property
	def n_init(self):
		return self._n_init
	@property
	def max_iter(self):
		return self._max_iter
	@property
	def tol(self):
		return self._tol
	@property
	def precompute_distances(self):
		return self._precompute_distances
	@property
	def verbose(self):
		return self._verbose
	@property
	def random_state(self):
		return self._random_state
	@property
	def copy_x(self):
		return self._copy_x
	@property
	def n_jobs(self):
		return self._n_jobs
	@property
	def algorithm(self):
		return self._algorithm
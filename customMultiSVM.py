from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
#Import a text feature extraction method from scikit. TfidVectorize applies both term 
#frequency and inverse document frequency to the input data.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
from random import randint
from scipy import sparse
import math

class data:
	
	def __init__(self):
		self.X_train = []
		self.y_train = {'toxic':[], 'severe_toxic':[], 'obscene':[], 'threat':[], 'insult':[], 'identity_hate':[]}
		self.X_test = []
		self.y_test = {'toxic':[], 'severe_toxic':[], 'obscene':[], 'threat':[], 'insult':[], 'identity_hate':[]}
	
	def set(self, filename):
		reader = csv.DictReader(open(filename), delimiter=',')
		X = []
		y = {'toxic':[], 'severe_toxic':[], 'obscene':[], 'threat':[], 'insult':[], 'identity_hate':[]}
		for row in reader:
			X.append(row['comment_text'])
			y['toxic'].append(float(row['toxic']))
			y['severe_toxic'].append(float(row['severe_toxic']))
			y['obscene'].append(float(row['obscene']))
			y['threat'].append(float(row['threat']))
			y['insult'].append(float(row['insult']))
			y['identity_hate'].append(float(row['identity_hate']))
		vectorizer = TfidfVectorizer()
		temp = vectorizer.fit_transform(X)
		self.X_train = temp[:int(temp.shape[0]*3/4), :]
		self.y_train['toxic'] = y['toxic'][:int(temp.shape[0]*3/4)]
		self.y_train['severe_toxic'] = y['severe_toxic'][:int(temp.shape[0]*3/4)]
		self.y_train['obscene'] = y['obscene'][:int(temp.shape[0]*3/4)]
		self.y_train['threat'] = y['threat'][:int(temp.shape[0]*3/4)]
		self.y_train['insult'] = y['insult'][:int(temp.shape[0]*3/4)]
		self.y_train['identity_hate'] = y['identity_hate'][:int(temp.shape[0]*3/4)]
		self.X_test = temp[int(temp.shape[0]*3/4):, :]
		self.y_test['toxic'] = y['toxic'][int(temp.shape[0]*3/4):]
		self.y_test['severe_toxic'] = y['severe_toxic'][int(temp.shape[0]*3/4):]
		self.y_test['obscene'] = y['obscene'][int(temp.shape[0]*3/4):]
		self.y_test['threat'] = y['threat'][int(temp.shape[0]*3/4):]
		self.y_test['insult'] = y['insult'][int(temp.shape[0]*3/4):]
		self.y_test['identity_hate'] = y['identity_hate'][int(temp.shape[0]*3/4):]
		
class multiSVM:
	def __init__(self):
		svmTox = LinearSVC()
		self.clfTox = CalibratedClassifierCV(svmTox)
		svmSev = LinearSVC()
		self.clfSev = CalibratedClassifierCV(svmSev)
		svmObc = LinearSVC()
		self.clfObc = CalibratedClassifierCV(svmObc)
		svmThr = LinearSVC()
		self.clfThr = CalibratedClassifierCV(svmThr)
		svmIns = LinearSVC()
		self.clfIns = CalibratedClassifierCV(svmIns)
		svmIde = LinearSVC()
		self.clfIde = CalibratedClassifierCV(svmIde)
	
	def fit(self, X, y):
		#X is passed in as a CRM and y is passed in as a dictionary 
		#with the keys being the different categories to be trained for
		#TODO: Switch to LinearSVC and try CalibratedClassifierCV to get probabilities
		
		print('Fitting 1')
		self.clfTox.fit(X,y['toxic']) 
		print('Fitting 2')
		self.clfSev.fit(X,y['severe_toxic'])
		print('Fitting 3')
		self.clfObc.fit(X,y['obscene'])
		print('Fitting 4')
		self.clfThr.fit(X,y['threat'])
		print('Fitting 5')
		self.clfIns.fit(X,y['insult'])
		print('Fitting 6')
		self.clfIde.fit(X,y['identity_hate'])

	def predict(self, X):
		y_pred = {'toxic':[], 'severe_toxic':[], 'obscene':[], 'threat':[], 'insult':[], 'identity_hate':[]}
		y_pred['toxic'] = self.clfTox.predict(X)
		y_pred['severe_toxic'] = self.clfSev.predict(X)
		y_pred['obscene'] = self.clfObc.predict(X)
		y_pred['threat'] = self.clfThr.predict(X)
		y_pred['insult'] = self.clfIns.predict(X)
		y_pred['identity_hate'] = self.clfIde.predict(X)
		return y_pred
		
def calcError(y_act, y_pred):
	error = 0
	for item in y_act:
		error += sum((y_act[item] - y_pred[item])**2)

	return error/len(y_act['toxic'])

def csrToSparseTensor(X):
	coo = X.tocoo()
	indices = np.mat([coo.row, coo.col]).transpose()
	return tf.SparseTensor(indices, coo.data, coo.shape)

class neuralSVM:
	def __init__(self):
		return












#Simple code to create a model based on the training file to get a rough esitmate of model's effectiveness
svm1 = multiSVM()

trainSet = data()
trainSet.set('train.csv')

svm1.fit(trainSet.X_train, trainSet.y_train)
y_pred = svm1.predict(trainSet.X_test)

error1 = calcError(trainSet.y_test, y_pred)
print(error1)
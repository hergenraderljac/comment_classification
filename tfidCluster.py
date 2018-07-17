import csv
#import tensorflow as tf
import numpy as np
from numpy import linalg as LA
#Import a text feature extraction method from scikit. TfidVectorize applies both term 
#frequency and inverse document frequency to the input data.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
from random import randint
from scipy import sparse
from sklearn import svm

#Function to read in a table a seperate it into the 
#id number and comment, does not read in any labels
def readTable(filename, id_Nums, comments):
	
	reader = csv.DictReader(open(filename), delimiter=',')
	
	for row in reader:
		id_Nums.append(row['id'])
		comments.append(row['comment_text'])

#Function to read in the labels of the given file if present
def readLabels(filename, labels):
	reader = csv.DictReader(open(filename), delimiter=',')
	for row in reader:
		labels.append([float(row['toxic']),float(row['severe_toxic']),float(row['obscene']),float(row['threat']),float(row['insult']),float(row['identity_hate'])])


class inputData:
	
	def __init__(self, filename):
		self.id_Nums = []
		self.comments = []
		readTable(filename, self.id_Nums, self.comments)
	
class inputLabels:
	
	def __init__(self, filename):
		self.labels = []
		readLabels(filename, self.labels)

class multiSVM:
	def __init__(self):
		self.svm1 = svm.SVC(probability=True)
		self.svm2 = svm.SVC(probability=True)
		self.svm3 = svm.SVC(probability=True)
		self.svm4 = svm.SVC(probability=True)
		self.svm5 = svm.SVC(probability=True)
		self.svm6 = svm.SVC(probability=True)
	
	def fit(self, X, y):
		print('Fitting 1')
		self.svm1.fit(X,y[:,1])
		print('Fitting 2')
		self.svm2.fit(X,y[:,2])
		print('Fitting 3')
		self.svm3.fit(X,y[:,3])
		print('Fitting 4')
		self.svm4.fit(X,y[:,4])
		print('Fitting 5')
		self.svm5.fit(X,y[:,5])
		print('Fitting 6')
		self.svm6.fit(X,y[:,6])
		




#documents is an object with comment text and comment id as member variables
documents = inputData('train.csv')
#y is an object with the labels as a member variable
y = inputLabels('train.csv')

X_train = []
X_test = []
y_train = []
y_test = []

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents.comments)

index = int(X.shape[0] * 0.75)
X_train = X[:index]
y_train = np.array(y.labels[:index])
X_test = X[index+1:]
y_test = np.array(y.labels[index+1:])

clf = OneVsRestClassifier(LinearSVC(random_state=0))
print('fitting')
clf.fit(sparse.csr_matrix(X_train),np.array(y_train))
y_pred = clf.predict(sparse.csr_matrix(X_test))


#Calculate total error?

dif = y_pred - np.array(y_test)

error_rate = 0
for row in dif:
	error_rate += LA.norm(row,1)
	
	
error_rate = error_rate/(y_pred.shape[0]*y_pred.shape[1])
print(error_rate)

















#This imports the SVM model from scikit's library
from sklearn import svm
#Import some basic datasets to mess around with into datasets
from sklearn import datasets
import pickle

clf = svm.SVC() #Create a svm model to be trained and assign it to clf

iris = datasets.load_iris() #Pull the iris dataset from the imported data sets
X, y = iris.data, iris.target #Load the training data and the label data into X and y respectively
clf.fit(X,y) #Train the svm model stored in clf using X as the training data and y as the labels

s = pickle.dumps(clf)
clf2 = pickle.loads(s)

print(clf.predict(X[0:1])) #Predict the output for the given input
print(y)

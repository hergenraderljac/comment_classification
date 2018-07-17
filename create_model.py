import numpy as np
import sklearn as skl
import pandas as pd
import pickle
import os.path



labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

class data:

	def __init__(self, filename, ratio=0.7):
		if not os.path.isfile(filename):
			print('Please use a valid filepath.')
			exit()
		self.rows = sum(1 for row in open(filename))

		chunksize = int(self.rows * .7)
		temp = pd.read_csv(filename)
		self.train_text = temp.comment_text[0:chunksize]
		self.train_labels = temp[labels][0:chunksize]
		self.val_text = temp.comment_text[chunksize:]
		self.val_labels = temp[labels][chunksize:]
		del temp
		return

	# TODO: Implement method to reset val and train sets 
	def reset(self, ratio=0.7):
		return None

def train_model(data=None):
	if data == None:
		print("That won't work! Try again with training data.")
		exit()
	



	






if __name__ == "__main__":
	test = data('train.csv')
	train_model(test)





# #Simple code to create a model based on the training file to get a rough esitmate of model's effectiveness
# svm1 = multiSVM()

# trainSet = data()
# trainSet.set('train.csv')

# svm1.fit(trainSet.X_train, trainSet.y_train)
# y_pred = svm1.predict(trainSet.X_test)

# error1 = calcError(trainSet.y_test, y_pred)
# print(error1)
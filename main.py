'''
Cicada Recognition Program
By: James Pirozzolo
Last Revision: 7/18/17

Purpose
-------
To be able to classify which of three classes a cicada, an insect, is from a picture.

Methods
-------
This python document is the main script, but it references the code to a MATLAB
program in order to extract required dimensions from an image.  In addition, it reads
data from an excel document as the training data for the machine learning classifier. 
The classifier used is a logistic regression program, which uses three-fold 
verification in order to perform its accuracy analysis.  

Accuracy
--------
The accuracy of the program hovers around 80%.  The program is very successful at 
evaluating images in which the insects are flat, but struggles when insects are on
their side.  It is more consistent with photos taken from above.  

Requirements
------------
Images need to have one of each type of cicada, in the foreground of the photo.  Their
order does not matter.  The output will give its classification in order of total area.





'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matlab.engine as me
import bisect

def plot_decision_regions(X, y, classifier,test_idx=None, resolution=0.02): 
	# setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
       plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
    
    # highlight test samples
    if test_idx:
       X_test, y_test = X[test_idx, :], y[test_idx]
       plt.scatter(X_test[:, 0], X_test[:, 1], c='',
               alpha=1.0, linewidth=1, marker='o',
               s=55, label='test set')

def extract(img):
	eng = me.start_matlab()
	eng.addpath('/Users/jamespirozzolo/Documents/MATLAB/Cicada', nargout=0)
	result = eng.extract(img, nargout = 1)
	ratios = []
	ratios2 = []

	np.array(result)
	result.shape = (3,2)
	values = result[0]

	for i in range(0,3):
		ratio = values[((-1*i)+2)]/values[2]
		ratios.append(ratio)
		ratio2 = values[2]/values[i]
		ratios2.append(ratio2)

	X_test = np.column_stack((ratios2, ratios))
	return X_test


def pre_processing(excel):
	df = pd.read_excel(excel)
	#reads dimensions for required ratios
	X = df.iloc[:, [11,12]].values
	y = df.iloc[:, 7].values

	return X, y


def plot(X_train_std, X_test_std, y_train, y_test, lr):

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X=X_combined_std,y=y_combined,classifier=lr,test_idx=range(0,32))
    plt.xlabel('Dimension 1 [standardized]')
    plt.ylabel('Dimension 2 [standardized]')
    plt.legend(loc='upper left')
    
    plt.show()

    

def main(img):
	X, y = pre_processing('/Users/jamespirozzolo/Documents/MATLAB/Cicada/CicadaDataSet.xlsx')

	accuracy=[]

	extracted_X = extract(img)
	
	#performs three-fold verification for accuracy analysis
	skf = StratifiedKFold(n_splits=3)
	for train, test in skf.split(X, y):
		
		#assigns variables
		X_train = X[train]
		X_test = X[test]
		y_train = y[train]
		y_test = y[test]

		#sets up pipeline, and fits pipeline to training data
		pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', LogisticRegression(random_state=1))])
		lr = LogisticRegression(C=1000.0, random_state=0)
		lr.fit(X_train, y_train)
		pipe_lr.fit(X_train, y_train)
		
		#compares to test data, and judges accuracy
		score = pipe_lr.score(X_test, y_test)
		accuracy.append(score)
		

	'''
	To plot the data or view the accuracy metric, uncomment the following lines:
	
	plot(X_train, X_test, y_train, y_test, lr)

	
	#gives average accuracy
	print(np.sum(accuracy)/3)
	'''


	#predicts extracted data
	extract_pred = pipe_lr.predict(extracted_X)


	
	
	print(extract_pred)


main()





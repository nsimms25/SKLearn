import numpy as np
import pandas as pd
import copy
import statistics
import math
import sklearn as sk
from sklearn import preprocessing, decomposition
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Function to normalize all data in *array
#Normalized range(0,1)
#Input: 2D NDArray (numpy 2D array) without and labels (only numeric values, error will rise else)
#Output print to a file called NormalizedMatrix.csv
def normalize(array):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    scaled_df = scaler.fit_transform(array)
    return scaled_df

train = pd.read_csv("TrainingDataSet.csv") #Read the training data set

#Remove label (True Class, Training Product label, and Data type (bool, int...))

TrueClass = train[['True Class']] # Save 'True Class' for later use in feature selection.
TrueClass = train.iloc[:, 26].apply(pd.Series)
TrueClass = pd.DataFrame(TrueClass)

#Remove True Class and Training Product label.
train = train.drop('True Class', 1)
train = train.drop(train.columns[0], 1)

featureNames = train.columns.values

#Polynomial Creation
polynomial = preprocessing.PolynomialFeatures(degree=2)

#Create 2D NDArray of type float64
NDTrain = np.array(train, dtype=np.float64)

#Call the normalization function.
scaled_train = normalize(NDTrain)
poly_train = polynomial.fit(scaled_train, TrueClass).transform(scaled_train)
polyPowers = polynomial.powers_
polyFeatures = polynomial.get_feature_names(featureNames)
poly_train = pd.DataFrame(poly_train, index=None)
poly_train.to_csv("PolynomialNormalizedTrain.csv")

#Feature Selection
select = SelectKBest(chi2, k=25).fit(poly_train, TrueClass)
featureSelect = select.transform(poly_train)
scores = select.scores_
scores = pd.DataFrame(scores, index = None)
scores.to_csv("SelectKBestScores.csv")

#print(featureSelect.shape)
featureSelect = pd.DataFrame(featureSelect)
featureSelect.to_csv("SelectKBest.csv")



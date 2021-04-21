import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#5x5

def convolve2D(image, filter):
  fX, fY = filter.shape 
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2) 
  newImage = np.zeros((nn,nn)) 
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage
train = pd.read_csv("train.csv")
X = train.drop('label',axis=1)
Y = train['label']
filter = np.array([[1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1]])
X = X.to_numpy()
print(X.shape)
sX = np.empty((0,576), int)
ss = 500 
for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolve2D(img2D,filter)
  nImg1D = np.reshape(nImg, (-1,576))
  sX = np.append(sX, nImg1D, axis=0)
Y = Y.to_numpy()
sY = Y[0:ss]
print(sY.shape)
print(sX.shape)
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
clf = MultinomialNB()
clf.fit(sXTrain, yTrain)
print(clf.class_count_)
print("5x5")
print(clf.score(sXTest, yTest))
print(sX.shape)





#7x7

def convolve2D(image, filter):
  fX, fY = filter.shape 
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2) 
  newImage = np.zeros((nn,nn)) 
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage
train = pd.read_csv("train.csv")
X = train.drop('label',axis=1)
Y = train['label']
filter = np.array([[1,1,1,1,1,1,1],
          [1,2,2,2,2,2,1],
          [1,2,3,3,3,2,1],
          [1,3,4,4,4,3,1],
          [1,2,3,3,3,2,1],
          [1,2,2,2,2,2,1],
          [1,1,1,1,1,1,1]])
X = X.to_numpy()
print(X.shape)
sX = np.empty((0,484), int)
ss = 500
for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolve2D(img2D,filter)
  nImg1D = np.reshape(nImg, (-1,484))
  sX = np.append(sX, nImg1D, axis=0)
Y = Y.to_numpy()
sY = Y[0:ss]
print(sY.shape)
print(sX.shape)
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
clf = MultinomialNB()
clf.fit(sXTrain, yTrain)
print(clf.class_count_)
print("7x7")
print(clf.score(sXTest, yTest))
print(sX.shape)




#9x9

def convolve2D(image, filter):
  fX, fY = filter.shape 
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2) 
  newImage = np.zeros((nn,nn)) 
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage
train = pd.read_csv("train.csv")
X = train.drop('label',axis=1)
Y = train['label']
filter = np.array([[1,1,1,1,1,1,1,1,1],
                   [1,2,2,2,2,2,2,2,1],
                   [1,2,2,3,3,3,2,2,1],
                   [1,2,2,3,3,2,2,2,1],
                   [1,2,3,4,4,4,3,2,1],
                   [1,2,2,3,3,3,2,2,1],
                   [1,2,2,3,2,3,2,2,1],
                   [1,2,2,2,2,2,2,2,1],
                   [1,1,1,1,1,1,1,1,1]
          ])

X = X.to_numpy()
print(X.shape)
sX = np.empty((0,400), int)
ss = 500 
for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolve2D(img2D,filter)
  nImg1D = np.reshape(nImg, (-1,400))
  sX = np.append(sX, nImg1D, axis=0)
Y = Y.to_numpy()
sY = Y[0:ss]
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
clf = MultinomialNB()
clf.fit(sXTrain, yTrain)
print(clf.class_count_)
print("9x9")
print(clf.score(sXTest, yTest))
print(sX.shape)
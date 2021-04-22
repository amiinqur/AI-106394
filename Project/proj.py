#multinomialNB Code with 3 convolution 
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
ss = 28000 
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

predict_test = clf.predict(sX)
out = zip(range(len(sX)), predict_test)
with open("Multimonial.csv", 'w') as g:
    g.write("ImageId,Label\n")
    for id, cat in out:
        g.write(str(id + 1) + "," + str(cat) + "\n")
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
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1]])

X = X.to_numpy()
print(X.shape)

sX = np.empty((0,484), int)


ss = 28000

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

print(clf.score(sXTest, yTest))

predict_test = clf.predict(sX)
out = zip(range(len(sX)), predict_test)
with open("Multimonial.csv", 'w') as g:
    g.write("ImageId,Label\n")
    for id, cat in out:
        g.write(str(id + 1) + "," + str(cat) + "\n")
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
ss = 28000 
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

predict_test = clf.predict(sX)
out = zip(range(len(sX)), predict_test)
with open("Multimonial.csv", 'w') as g:
    g.write("ImageId,Label\n")
    for id, cat in out:
        g.write(str(id + 1) + "," + str(cat) + "\n")
print(sX.shape)


#KNN WITH 3 convolution
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



#5X5 KNN
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


ss = 28000 

for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))

  nImg = convolve2D(img2D,filter)

  nImg1D = np.reshape(nImg, (-1,576))

  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]

print(sY.shape)
print(sX.shape)


X_train, X_test, Y_train, Y_test = train_test_split(sX,sY,test_size=0.1)
scaler= StandardScaler()
X_train = scaler.fit_transform(X_train)
print(X_test.shape)
X_test = scaler.fit_transform(X_test)

print('Length: ',len(Y_test))
print('K: ',np.sqrt(len(Y_test)))

classifier = KNeighborsClassifier(n_neighbors=7,p=2,metric='euclidean')
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
print(classification_report(Y_test,Y_pred))
print("Accuracy --> ", end= ' ')
print(accuracy_score(Y_test,Y_pred)*100)
predict_test = classifier.predict(sX)
out = zip(range(len(sX)), predict_test)
with open("KNN.csv", 'w') as g:
    g.write("ImageId,Label\n")
    for id, cat in out:
        g.write(str(id + 1) + "," + str(cat) + "\n")

        
#7X7 KNN
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
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1]])

X = X.to_numpy()
print(X.shape)

sX = np.empty((0,484), int)


ss = 28000 

for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))

  nImg = convolve2D(img2D,filter)

  nImg1D = np.reshape(nImg, (-1,484))

  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]

print(sY.shape)
print(sX.shape)


X_train, X_test, Y_train, Y_test = train_test_split(sX,sY,test_size=0.1)
scaler= StandardScaler()
X_train = scaler.fit_transform(X_train)
print(X_test.shape)
X_test = scaler.fit_transform(X_test)

print('Length: ',len(Y_test))
print('K: ',np.sqrt(len(Y_test)))

classifier = KNeighborsClassifier(n_neighbors=7,p=2,metric='euclidean')
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
print(classification_report(Y_test,Y_pred))
print("Accuracy --> ", end= ' ')
print(accuracy_score(Y_test,Y_pred)*100)
predict_test = classifier.predict(sX)
out = zip(range(len(sX)), predict_test)
with open("KNN.csv", 'w') as g:
    g.write("ImageId,Label\n")
    for id, cat in out:
        g.write(str(id + 1) + "," + str(cat) + "\n")
#9X9



def convolve2D(image, filter):
  fX, fY = filter.shape # Get filter dimensions
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2) #new dimension of the reduced image size
  newImage = np.zeros((nn,nn)) #empty new 2D imange
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

# img = X[6]
ss = 2800 


for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))

  nImg = convolve2D(img2D,filter)
  nImg1D = np.reshape(nImg, (-1,400))
  # print(nImg.shape)
  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]
# print(sY)
print(sY.shape)
print(sX.shape)

X_train, X_test, Y_train, Y_test = train_test_split(sX,sY,test_size=0.1)
scaler= StandardScaler()
X_train = scaler.fit_transform(X_train)
print(X_test.shape)
X_test = scaler.fit_transform(X_test)

print('Length: ',len(Y_test))
print('K: ',np.sqrt(len(Y_test)))

classifier = KNeighborsClassifier(n_neighbors=7,p=2,metric='euclidean')
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
print(classification_report(Y_test,Y_pred))
print("Accuracy --> ", end= ' ')
print(accuracy_score(Y_test,Y_pred)*100)
predict_test = classifier.predict(sX)
out = zip(range(len(sX)), predict_test)
with open("KNN.csv", 'w') as g:
    g.write("ImageId,Label\n")
    for id, cat in out:
        g.write(str(id + 1) + "," + str(cat) + "\n")
#SVM WITH 3 CONVOLUTIONS
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#Convolution 5X5
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


ss = 28000 

for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))

  nImg = convolve2D(img2D,filter)

  nImg1D = np.reshape(nImg, (-1,576))

  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]

print(sY.shape)
print(sX.shape)

X_train, X_test, Y_train, Y_test = train_test_split(sX,sY,test_size=0.20,random_state=1)
svcClassifier=SVC(kernel='linear')
svcClassifier.fit(X_train,Y_train)
Y_pred = svcClassifier.predict(X_test)
print('Accuracy Linear: ',accuracy_score(Y_test,Y_pred)*100)

svcClassifier=SVC(kernel='rbf')
svcClassifier.fit(X_train,Y_train)
Y_pred = svcClassifier.predict(X_test)
print('Accuracy RBF: ',accuracy_score(Y_test,Y_pred)*100)


parameters={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}
grid=GridSearchCV(SVC(),parameters,verbose=0)
grid.fit(X_train,Y_train)
print('Best Parameters: ',grid.best_params_)
gridPred = grid.predict(X_test)
print('Accuracy Parameters: ',accuracy_score(Y_test,gridPred)*100)
predict_test = svcClassifier.predict(sX)
out = zip(range(len(sX)), predict_test)
with open("SVM.csv", 'w') as g:
    g.write("ImageId,Label\n")
    for id, cat in out:
        g.write(str(id + 1) + "," + str(cat) + "\n")
#7X7 Convolution
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
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1]])


X = X.to_numpy()
print(X.shape)

sX = np.empty((0,484), int)


ss = 28000 

for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))

  nImg = convolve2D(img2D,filter)

  nImg1D = np.reshape(nImg, (-1,484))

  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]

X_train, X_test, Y_train, Y_test = train_test_split(sX,sY,test_size=0.20,random_state=1)
svcClassifier=SVC(kernel='linear')
svcClassifier.fit(X_train,Y_train)
Y_pred = svcClassifier.predict(X_test)
print('Accuracy Linear: ',accuracy_score(Y_test,Y_pred)*100)

svcClassifier=SVC(kernel='rbf')
svcClassifier.fit(X_train,Y_train)
Y_pred = svcClassifier.predict(X_test)
print('Accuracy RBF: ',accuracy_score(Y_test,Y_pred)*100)


parameters={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}
grid=GridSearchCV(SVC(),parameters,verbose=0)
grid.fit(X_train,Y_train)
print('Best Parameters: ',grid.best_params_)
gridPred = grid.predict(X_test)
print('Accuracy Parameters: ',accuracy_score(Y_test,gridPred)*100)
predict_test = svcClassifier.predict(sX)
out = zip(range(len(sX)), predict_test)
with open("SVM.csv", 'w') as g:
    g.write("ImageId,Label\n")
    for id, cat in out:
        g.write(str(id + 1) + "," + str(cat) + "\n")
#9X9 CONVOLUTION
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


ss = 28000 

for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))

  nImg = convolve2D(img2D,filter)

  nImg1D = np.reshape(nImg, (-1,400))

  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]

X_train, X_test, Y_train, Y_test = train_test_split(sX,sY,test_size=0.20,random_state=1)
svcClassifier=SVC(kernel='linear')
svcClassifier.fit(X_train,Y_train)
Y_pred = svcClassifier.predict(X_test)
print('Accuracy Linear: ',accuracy_score(Y_test,Y_pred)*100)

svcClassifier=SVC(kernel='rbf')
svcClassifier.fit(X_train,Y_train)
Y_pred = svcClassifier.predict(X_test)
print('Accuracy RBF: ',accuracy_score(Y_test,Y_pred)*100)


parameters={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}
grid=GridSearchCV(SVC(),parameters,verbose=0)
grid.fit(X_train,Y_train)
print('Best Parameters: ',grid.best_params_)
gridPred = grid.predict(X_test)
print('Accuracy Parameters: ',accuracy_score(Y_test,gridPred)*100)
predict_test = svcClassifier.predict(sX)
out = zip(range(len(sX)), predict_test)
with open("SVM.csv", 'w') as g:
    g.write("ImageId,Label\n")
    for id, cat in out:
        g.write(str(id + 1) + "," + str(cat) + "\n")

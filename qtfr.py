import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from joblib import dump, load
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the data
dataframe1 = pd.read_excel("Object1.xlsx", header= None)
data1 = np.array(dataframe1.drop([0,1,2,3,4,5], axis=1))

dataframe2 = pd.read_excel("NotObject1.xlsx", header= None)
data2 = np.array(dataframe2.drop([0,1,2,3,4,5], axis=1))

# Feature Extraction
n_features = 3

X1 = []
for i in range(data1.shape[0]):
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(data1[i], Fs=1)
    plt.close()
    X1.append([np.sum(powerSpectrum),np.argmax(powerSpectrum),np.where(powerSpectrum == np.amax(powerSpectrum))[1][0]])

X2 = []
for i in range(data2.shape[0]):
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(data2[i], Fs=1)
    plt.close()
    X2.append([np.sum(powerSpectrum),np.argmax(powerSpectrum),np.where(powerSpectrum == np.amax(powerSpectrum))[1][0]])

X1 = np.array(X1)
X1 = X1.reshape(X1.shape[0],n_features)

X2 = np.array(X2)
X2 = X2.reshape(X2.shape[0],n_features)

X = np.vstack((X1,X2)) # Features
Y = np.hstack((np.zeros(X1.shape[0]),np.ones(X2.shape[0]))) # Labels

# Train and Test Splitting
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Train the model
nbc = GaussianNB()
nbc.fit(x_train, y_train)
    
# Save the Model
dump(nbc,'bayes_classifier_model.joblib')

# Load the Model
nbc_loaded = load('bayes_classifier_model.joblib')

# Predict using the loaded model
y_pred = nbc_loaded.predict(x_test)

# Test the model after training
print('Accuracy:',metrics.accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred).ravel()
tn, fp, fn, tp  = cm
disp = ConfusionMatrixDisplay(confusion_matrix=cm.reshape(2,2))
disp.plot()

tpr = tp/(tp + fn)
tnr = tn/(tn + fp)
fdr = fp/(fp + tp)
npv = tn/(tn + fn)

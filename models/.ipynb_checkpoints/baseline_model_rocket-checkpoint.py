import numpy as np
import pandas as pd
import torch #as th
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

path = os.getcwd() #'/Users/fuad/Documents/GitHub/models'
last_indx = path.rindex("/")

# set project folder location where the models and data folder exist
projectlocation = path[0:last_indx]# '/Users/fuad/Documents/GitHub'

#set data path and load data
datapath = projectlocation + "/data/flare_prediction_mvts_data.pck"
labelpath = projectlocation + "/data/flare_prediction_labels.pck"

mvts_1540=load(datapath)
labels_1540=load(labelpath)

#Check data
print("MVTS dataset shape: ", mvts_1540.shape, "  type: ", type(mvts_1540)) # (1540, 33, 60)
print("Labels shape: ", labels_1540.shape, "  type: ", type(labels_1540)) 

#Binary classification -->label conversion to BINARY class
def get_binary_labels_from(labels_str):
    tdf = pd.DataFrame(labels_str, columns = ['labels'])
    data_classes= [0, 1, 2, 3]
    d = dict(zip(data_classes, [0, 0, 1, 1])) 
    arr = tdf['labels'].map(d, na_action='ignore')
    return arr.to_numpy()

#uncomment next line for binary classification
#labels_1540 = get_binary_labels_from(labels_1540)


#Takes 3D array(x,y,z) >> transpose(y,z) >> return (x,z,y)
def GetTransposed2D(arrayFrom):
    toReturn = []
    alen = arrayFrom.shape[0]
    for i in range(0, alen):
        toReturn.append(arrayFrom[i].T)
    
    return np.array(toReturn)

#Takes 3D array(x,y,z) >> Flatten() >> return (y,z)
def Make2D(array3D):
    toReturn = []
    x = array3D.shape[0]
    y = array3D.shape[1]
    for i in range(0, x):
        for j in range(0, y):
            toReturn.append(array3D[i,j])
    
    return np.array(toReturn)

#Transform instance(92400, 33) into(1540x60x33)
def Get3D_MVTS_from2D(array2D, windowSize):
    arrlen = array2D.shape[0]
    mvts = []
    for i in range(0, arrlen, windowSize):
        mvts.append(array2D[i:i+windowSize])

    return np.array(mvts)

from sklearn.preprocessing import StandardScaler
#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
def GetStandardScaler(data2d):
    scaler = StandardScaler()
    scaler = scaler.fit(data2d)
    return scaler
def GetStandardScaledData(data2d):
    scaler = StandardScaler()
    scaler = scaler.fit(data2d)
    #print(scaler.mean_)
    data_scaled = scaler.transform(data2d)
    return data_scaled

def transform_scale_data(data3d, scaler):
    #print("original data shape:", data3d.shape) 
    trans = GetTransposed2D(data3d)
    #print("transposed data shape:", trans.shape)    #(x, 60, 33)
    data2d = Make2D(trans)
    #print("2d data shape:", data2d.shape)    
    #  scaler = GetStandardScaler(data2d)
    data_scaled = scaler.transform(data2d)
    mvts_scalled = Get3D_MVTS_from2D(data_scaled, data3d.shape[2])#,60)
    #print("mvts data shape:", mvts_scalled.shape)
    transBack = GetTransposed2D(mvts_scalled)
    #print("transBack data shape:", transBack.shape)
    return transBack


from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score


def doClassSpecificCalulcation(classification_report_dict, y_test):
    
  Accuracy = []
  for i in range(len(classification_report_dict)):
      report=classification_report_dict[i]
      temp=report['accuracy']
      Accuracy.append(temp)

  print('mean(Accuracy) :',np.mean(Accuracy))
  print('std(Accuracy) :',np.std(Accuracy))
  print('mean np.std(Accuracy) : ',np.round(np.mean(Accuracy),2),"+-",np.round(np.std(Accuracy),2) )

  for j in np.unique(y_test):#np.unique(trainLebel ): #. len(...) np.unique(trainLebel):  [0 1 2 3]
    print('\nclass :',j) 
    precision=[]
    recall=[]
    f1_score=[]
    for i in range(len(classification_report_dict)):
      report=classification_report_dict[i]
      #print('classification_report : \n',report) 
      temp=report[str(j)]['precision'] 
      precision.append(temp)

      temp=report[str(j)]['recall'] 
      recall.append(temp)

      temp=report[str(j)]['f1-score'] 
      f1_score.append(temp)

    print('p.mean(precision) \t p.mean(recall) \t p.mean(f1_score) :') 
    print(np.mean(precision), "\t", np.mean(recall), "\t", np.mean(f1_score)) 
    #print(np.mean(recall)) 
    #print(np.mean(f1_score))

    print('p.mean p.std(precision) \tp.mean  p.std(recall) \tp.mean  p.std(f1_score) :')
    print(np.round(np.mean(precision),2),"+-",np.round(np.std(precision),2) )
    print(np.round(np.mean(recall),2),"+-",np.round(np.std(recall),2) )
    print(np.round(np.mean(f1_score),2),"+-",np.round(np.std(f1_score),2) )

"""**Rocket**"""

#!pip install sktime # OR #!pip install 'sktime[all_extras]'
#!pip install --upgrade numba 
#-----> ROCKET compiles (via Numba) on import, which may take a few seconds.
from sklearn.linear_model import RidgeClassifierCV
from sktime.transformations.panel.rocket import Rocket

def get_rocket_predictions(Xtrain, ytrain, Xtest):
    rocket = Rocket()
    rocket.fit(Xtrain)
    Xtrain_transform = rocket.transform(Xtrain)
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    classifier.fit(Xtrain_transform, ytrain)
    
    Xtest_transform = rocket.transform(Xtest)
    preds = classifier.predict(Xtest_transform)
    return preds

# run experiment of 5 differeent random_state of train_test_data_split
# experiment with different train size (0.1 to 0.9); change test_size
classification_report_dict=[]
for i in range(0,1):
    print("experiment running with random_state = ", i, " ...")
    X_train, X_test, y_train, y_test = train_test_split(
        mvts_1540, labels_1540, test_size=0.3, random_state = i, stratify=labels_1540)

    trans = GetTransposed2D(X_train)
    data2d = Make2D(trans)
    scaler = GetStandardScaler(data2d)

    X_train = transform_scale_data(X_train, scaler)
    X_test = transform_scale_data(X_test, scaler)

    predictions = get_rocket_predictions(X_train, y_train, X_test)
    report_dict = metrics.classification_report(y_test, predictions, digits=3,output_dict=True)
    classification_report_dict.append(report_dict)

doClassSpecificCalulcation(classification_report_dict, y_test)
import json
import pandas as pd
import numpy as np
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import scipy.stats as st
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

# Data loading from local file
import pickle
import os

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

path = os.getcwd() #'/Users/fuad/Documents/GitHub/models'

# set project folder location where the models and data folder exist
projectlocation = '/Users/fuad/Documents/GitHub'

#set data path and load data
datapath = projectlocation + "/data/flare_prediction_mvts_data.pck"
labelpath = projectlocation + "/data/flare_prediction_labels.pck"

trainData = load(datapath)
trainLebel = load(labelpath)

print("trainData.shape: ",trainData.shape)
print("trainLebel.shape: ",trainLebel.shape)

#four-class problem {X, M, B/C, Q}
print("np.unique(trainLebel): ",np.unique(trainLebel))

#Binary classification
# Data label conversion to BINARY class
def get_binary_labels_from(labels_str):
    tdf = pd.DataFrame(labels_str, columns = ['labels'])
    data_classes= [0, 1, 2, 3]
    d = dict(zip(data_classes, [0, 0, 1, 1])) 
    arr = tdf['labels'].map(d, na_action='ignore')
    return arr.to_numpy()

#un-comment next line for Binary classification experiment
trainLebel = get_binary_labels_from(trainLebel)

temptrainData=np.empty([1540,60, 33])
n=len(trainData)
for l in range(0, n):
  temp=trainData[l]
  #print(temp)
  #temp=np.transpose(temp)
  temp=temp.T
  #print(temp.shape)
  #print(temp)
  temptrainData[l,:,:]=temp
  n=n+1
  #np.append(temptrainData, temp)
  #print(temptrainData)

#print(temptrainData.shape)
#print(trainData.shape) 
trainData=temptrainData
print("trainData.shape: ",trainData.shape)
#print(trainData[0])

temp=trainData[0]
#print(temp)
df = pd.DataFrame(temp)
#df=pd.DataFrame.from_dict(trainData)
trainData222=trainData

df

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
print(trainData.shape)

print(type(trainData))
npArrays=[]
for l in range(0, len(trainData)):
  trainData_std = sc.fit_transform(trainData[l])
  npArrays.append(trainData_std)

print(type(npArrays))
arr = np.asarray(npArrays)
print(type(arr))
trainData_scaled=arr
df = pd.DataFrame(trainData_scaled[0])

num_mvts=trainData_scaled.shape[0]
num_ts=trainData_scaled.shape[1]
num_params=trainData_scaled.shape[2]

#num_mvts=2
#num_mvts=len(trainData_scaled)
x = np.zeros( (num_mvts,num_params*8) )
print(x.shape)

for i in range(0, num_mvts):
  #print("mvts: ",i)
  temp=trainData_scaled[i]
  

  temp_X = np.zeros( (num_params,8) )

  for j in range(0, num_params):
    #print("param: ",j)
    ts = trainData_scaled[i,:, j]
    #print("ts length: ",ts.shape)

    v1 = np.mean(ts)
    v2 = np.std(ts)
    v3 = st.skew(ts)
    v4 = st.kurtosis(ts)

    ts_p = np.zeros((len(ts) - 1))
    for k in range(len(ts_p)):
       ts_p[k] = ts[k + 1] - ts[k]
       
    v5 = np.mean(ts_p)
    v6 = np.std(ts_p)
    v7 = st.skew(ts_p)
    v8 = st.kurtosis(ts_p)
    vect_ts = [v1, v2, v3, v4, v5, v6, v7, v8]
    #X[i, j * 8:j * 8 + 8] = vect_ts
    #print(col, " vect_ts: ",vect_ts)
    temp_X[j,:] = vect_ts
    #print("x[",col,"]=",X[col])
  #print("temp_X: ",temp_X.shape)
  #print(X)

  row = temp_X.reshape((num_params*8))
  x[i,:]=row

  #x2=temp_X[0]
  #for row in range(1, colNumber):
    #x2=np.concatenate((x2,temp_X[1]), axis=0)
  #print("x2.shape: ",x2.shape)
  #x[table] = x2

trainData_8n=x
#print("x.shape: ",x.shape) 
print("trainData_8n.shape: ",trainData_8n.shape)

temp=trainData_8n[0]
#print(temp)
df = pd.DataFrame(temp)
#df=pd.DataFrame.from_dict(trainData_8n)

df

import random

from sklearn.metrics import roc_auc_score

def doBaseLineMethodsBasedCalcultions(inputData, X_train, X_test, y_train, y_test):
  num_masterIteration=5

  #inputData=trainData_8n

  #print(type(inputData))
  #print(type(inputData[0]))
  #print(inputData.shape)

  classification_report_dict=[]
  #Accuracy=[][]
  Accuracy=[]
  roc_auc=[]
  for masterIteration in range(num_masterIteration):
      print("\nmasterIteration: ",masterIteration)
      #print(bcolors.WARNING + "\nmasterIteration :" + bcolors.WARNING,masterIteration)
      svm_model = SVC()
      rf_model = RandomForestClassifier()
      #forest = RandomForestClassifier(criterion='entropy',n_estimators=10, random_state=1, n_jobs=2)
      nb_model = GaussianNB()
      dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
      knn = neighbors.KNeighborsClassifier(n_neighbors=2)

      # all parameters not specified are set to their defaults
      #logisticRegr = LogisticRegression()
      #logisticRegr = LogisticRegression(max_iter=1000)
      #https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
      logisticRegr = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=4000)

      #tree = DecisionTreeClassifier( random_state=23)
      #adaboost = AdaBoostClassifier(base_estimator=tree, n_estimators=5, learning_rate=0.1, random_state=23)
      #bagging = BaggingClassifier(base_estimator=tree, n_estimators=5, max_samples=50, bootstrap=True)

      models = [svm_model, rf_model, nb_model,dt,knn]
      #models = [logisticRegr]
      for i in range(len(models)):
          mod = models[i]
          model_name = type(mod).__name__
          #print( model_name)
          mod.fit(X_train, y_train)

          #print(f"Train score: {mod.score(X_train, y_train)}")
          #print(f"Test score: {mod.score(X_test, y_test)}")

          y_pred = mod.predict(X_test)
          aaafsfsfsf=accuracy_score(y_test, y_pred)
          #print(model_name, ' Accuracy: %.4f' %aaafsfsfsf )
          Accuracy.append(aaafsfsfsf)

          sfsgsg=roc_auc_score(y_test, y_pred)
          roc_auc.append(sfsgsg)
          #print(model_name, ' roc_auc: %.4f' %sfsgsg )
          #print('Adjusted Accuracy : %.3f' % adjusted_rand_score(labels_true=y_test, labels_pred=y_pred))
          #print("classification_report:\n ", classification_report(y_test, y_pred))
        

          #TN, FP, FN, TP = 
          confusion_matrix=metrics.confusion_matrix(y_test, y_pred,labels=np.unique(trainLebel))
          #print(model_name,'Confusion Matrix : \n', confusion_matrix)
          #print(type(y_test))
          #print(type(y_pred))
          sfsfsf2=metrics.classification_report(y_test, y_pred, digits=3)
  
          print(model_name,'classification_report : \n',sfsfsf2)

          sfsfsf=metrics.classification_report(y_test, y_pred, digits=3,output_dict=True)
          classification_report_dict.append(sfsfsf)
          
          #print(model_name,'classification_report_dict : \n',classification_report_dict)
      #print('roc_auc : \n',roc_auc)
  doClassSpecificCalulcation(roc_auc,Accuracy,trainLebel,classification_report_dict)

def doClassSpecificCalulcation(roc_auc,Accuracy,trainLebel,classification_report_dict):
  print("\n\n\n *************** Final Report ***************")
  print('\np.mean(roc_auc) :',np.mean(roc_auc))
  print('\np.std(roc_auc) :',np.std(roc_auc))
  print('\np.mean(Accuracy) :',np.mean(Accuracy))
  print('\np.std(Accuracy) :',np.std(Accuracy))
  print('\n33333333 p.mean np.std(Accuracy) :     ',np.round(np.mean(Accuracy),2),"+-",np.std(Accuracy) )
  for j in range( len(np.unique(trainLebel)) ):
    print('\n\n\n\nclass :',j) 
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

    print('\np.mean(precision) \t p.mean(recall) \t p.mean(f1_score) :') 


    print(np.mean(precision)) 
    print(np.mean(recall)) 
    print(np.mean(f1_score))

    print('\np.mean p.std(precision) \tp.mean  p.std(recall) \tp.mean  p.std(f1_score) :')

    print(np.round(np.mean(precision),2),"+-",np.std(precision) )
    print(np.round(np.mean(recall),2),"+-",np.std(recall) )
    print(np.round(np.mean(f1_score),2),"+-",np.std(f1_score) )

print(trainData_scaled.shape)
n=len(trainData_scaled)
lastRows=[]
for l in range(0, n):
  temp = trainData_scaled[l]
  #print(temp.shape)
  lastROw=temp[trainData_scaled.shape[1]-1]
  #print(lastROw.shape)
  #print(lastROw)
  #print(type(lastROw))
  lastRows.append(lastROw)

#print(lastRows)
df_lastRows = pd.DataFrame(lastRows)

lastRows = np.array(lastRows)

print(lastRows.shape)

print(trainData_scaled.shape)
n=len(trainData_scaled)
flatten_tables=[]
for l in range(0, n):
  temp = trainData_scaled[l]
  flatten_temp=temp.ravel()
  #print(flatten_temp.shape)
  #print(flatten_temp)
  #print(type(flatten_temp))
  flatten_tables.append(flatten_temp)

df_flatten_tables = pd.DataFrame(flatten_tables)

flatten_tables = np.array(flatten_tables)

print(flatten_tables.shape)

def startCalculations(inputData):
  #test_sizes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
  test_sizes=[0.3]
  for temp4 in range(len(test_sizes)):
    test_size=test_sizes[temp4]
    print("\n\n\n *************** test_size: ",test_size)
    random_state=0#random.randint(42, 100)
    print("random_state: ",random_state)
    X_train, X_test, y_train, y_test = train_test_split(inputData, trainLebel, 
                                                            test_size=test_size, 
                                                            random_state=random_state)
    print("X_train.shape X_test.shape y_train.shape y_test.shape ",
              X_train.shape, X_test.shape ,y_train.shape, y_test.shape)
    doBaseLineMethodsBasedCalcultions(inputData, X_train, X_test, y_train, y_test)

startCalculations(lastRows)##(1540, 33)

startCalculations(trainData_8n)

startCalculations(df_flatten_tables)


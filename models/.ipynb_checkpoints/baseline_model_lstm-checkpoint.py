import json
import pandas as pd
import numpy as np
from numpy import array
from numpy import hstack

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

from datetime import datetime
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score

#import matplotlib.pyplot as plt

from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
trainData=temptrainData
print("trainData.shape: ",trainData.shape)
#print(trainData[0])

temp=trainData[0]
#print(temp)
df = pd.DataFrame(temp)
#df=pd.DataFrame.from_dict(trainData)
trainData222=trainData

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
print(trainData.shape)

print(type(trainData))
npArrays=[]
for l in range(0, len(trainData)):
  trainData_std = sc.fit_transform(trainData[l])
  #trainData_std = trainData_std.astype(np.float64)
  #print(type(trainData_std[0][0]))
  npArrays.append(trainData_std)

print(type(npArrays))
arr = np.asarray(npArrays)
print(type(arr))
trainData=arr
df = pd.DataFrame(trainData[0])

INPUT_DIM = 33
HIDDEN_DIM = 64
NUM_TS = 60
NUM_CLASSES = 4
num_layers = 1 #number of stacked lstm layers
hidden_size=HIDDEN_DIM

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#input_dim = 33, hidden_dim = 128, num_TS = 60, num_classes = 4
class LSTM_MVTS_LRN(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_classes):
    super(LSTM_MVTS_LRN, self).__init__()
    self.hidden_dim = hidden_dim
    
    self.lstm = nn.LSTM(input_dim, hidden_dim)
    #self.lstm = nn.RNN(input_dim, hidden_dim)
    self.hidden2class = nn.Linear(hidden_dim, num_classes)
  def forward(self, mvts):
    print("model():","mvts.shape: ",mvts.shape)
    #print(mvts.shape)
    #input single mvts (60, 33); output class probability vector (1,4)
    lstm_out, _ = self.lstm(mvts.view(len(mvts), 1, -1)) #mvts.shape: (60, 33); len(mvts)=60; new shape: (60, 1, 33); lstm_out --> (60, 128)
    last_lstm_out = lstm_out[len(lstm_out)-1] #(1,128)
    class_space = self.hidden2class(last_lstm_out) #(1,4)
    class_scores = F.log_softmax(class_space, dim=1)
    return class_scores

import random
import matplotlib.pyplot as plt

def doLstmBasedCalculations( X_train, X_test, y_train, y_test,hds):
    HIDDEN_DIM=hds
    num_masterIteration=1

    classification_report_dict=[]
    Accuracy=[]
    for masterIteration in range(num_masterIteration):
        print("\nmasterIteration HIDDEN_DIM : ",masterIteration, HIDDEN_DIM)

        model = LSTM_MVTS_LRN(INPUT_DIM, 
                              #hds,
                              HIDDEN_DIM, 
                              NUM_CLASSES)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        numTrain = X_train.shape[0]

        numEpochs =500
        numEpochs =5

        #train
        for epoch in range(numEpochs):
          print("\n nmasterIteration, epoch: ",masterIteration,epoch)
          loss_values = []
          running_loss = 0.0


          for i in range(numTrain):
            model.zero_grad()
            mvts = X_train[i,:,:]
            mvts = torch.from_numpy(mvts).float()


            target = y_train[i]
            target = [target]
            target=np.array(target)
            target = torch.Tensor(target)
            target = target.type(torch.LongTensor)
            mvts = mvts.to(device)
            target = target.to(device)
            
            mvts = mvts.view(mvts.size(0), -1)

            model.to(device)
            print("mvts no:",i ,"mvts.shape: ",mvts.shape)
            class_scores = model(mvts)
            loss = loss_function(class_scores, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            #loss_values.append(running_loss)
            loss_values.append(running_loss / len(X_train))

          maxAcc=0
          max_classification_report_dict=0
          #print("loss: ",loss)  
          #test 
          numTest = X_test.shape[0]
          with torch.no_grad():
              numCorrect = 0
              testLabel=[]
              predictaedLabel=[]
              for i in range(numTest):
                test_mvts = X_test[i,:,:]
                test_label = y_test[i] #class = 2

                #model.zero_grad()

                test_mvts = torch.from_numpy(test_mvts).float()

                test_mvts = test_mvts.to(device)
                test_class_scores = model(test_mvts) 
                class_prediction = torch.argmax(test_class_scores, dim=-1) 
                current_seq = np.argmax(test_class_scores.cpu().numpy())
                testLabel.append(test_label)
                predictaedLabel.append(current_seq)



                if(class_prediction == test_label): #(2,3 ) match 
                  numCorrect = numCorrect+1
              acc = numCorrect/numTest

              fgdg=round(acc, 2)
              #print("fgdg:", fgdg) 
              if fgdg  > maxAcc:
                maxAcc=acc
                print(bcolors.WARNING + "maxAcc:" + bcolors.ENDC,maxAcc)
                max_classification_report_dict=metrics.classification_report(testLabel, predictaedLabel, digits=3,output_dict=True)

        plt.plot(np.array(loss_values), 'r')
        classification_report_dict.append(max_classification_report_dict)   
        #print('classification_report_dict : \n',classification_report_dict)
        Accuracy.append(maxAcc)  
        #print('Accuracy : \n',Accuracy)
    doClassSpecificCalulcation(Accuracy,trainLebel,classification_report_dict)

def doClassSpecificCalulcation(Accuracy,trainLebel,classification_report_dict):
  print('\np.mean(Accuracy) :',np.mean(Accuracy))
  print('\np.std(Accuracy) :',np.std(Accuracy))
  print('\n33333333 p.mean np.std(Accuracy) :     ',np.round(np.mean(Accuracy),2),"+-",np.round(np.std(Accuracy),2) )
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

    print(np.round(np.mean(precision),2),"+-",np.round(np.std(precision),2) )
    print(np.round(np.mean(recall),2),"+-",np.round(np.std(recall),2) )
    print(np.round(np.mean(f1_score),2),"+-",np.round(np.std(f1_score),2) )

def startCalulations():

  #test_sizes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
  test_sizes=[0.3]
  
  for temp4 in range(len(test_sizes)):
    test_size=test_sizes[temp4]
    print("\n\n\n *************** test_size: ",test_size)
    random_state=0#random.randint(42, 100)
    print("random_state: ",random_state)
    
    X_train, X_test, y_train, y_test = train_test_split(trainData, trainLebel, test_size=test_size, random_state=random_state)
   
    print("X_train.shape X_test.shape y_train.shape y_test.shape ",
              X_train.shape, X_test.shape ,y_train.shape, y_test.shape)
    
    HIDDEN_DIMs=[32,64,96,128,160,192,224,256,512]
    HIDDEN_DIMs=[32]
    for temp5 in range(len(HIDDEN_DIMs)):
        hds=HIDDEN_DIMs[temp5]
        doLstmBasedCalculations( X_train, X_test, y_train, y_test,hds)

startCalulations()


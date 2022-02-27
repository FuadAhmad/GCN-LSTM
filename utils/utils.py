import numpy as np
import pandas as pd
import torch #as th
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

#Load data
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

#Binary classification --> label conversion to BINARY class
def get_binary_labels_from(labels_str):
    tdf = pd.DataFrame(labels_str, columns = ['labels'])
    data_classes= [0, 1, 2, 3]
    d = dict(zip(data_classes, [0, 0, 1, 1])) 
    arr = tdf['labels'].map(d, na_action='ignore')
    return arr.to_numpy()

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
    print("original data shape:", data3d.shape) 
    trans = GetTransposed2D(data3d)
    print("transposed data shape:", trans.shape)    #(x, 60, 33)
    data2d = Make2D(trans)
    print("2d data shape:", data2d.shape)    
    #  scaler = GetStandardScaler(data2d)
    data_scaled = scaler.transform(data2d)
    mvts_scalled = Get3D_MVTS_from2D(data_scaled, data3d.shape[2])#,60)
    print("mvts data shape:", mvts_scalled.shape)
    transBack = GetTransposed2D(mvts_scalled)
    print("transBack data shape:", transBack.shape)
    return transBack

def build_edge_index_tensor(adj):
  num_nodes = adj.shape[0]
  source_nodes_ids, target_nodes_ids = [], []
  for i in range(num_nodes):
    for j in range(num_nodes):
      if(adj[i,j]==1):
        source_nodes_ids.append(i)
        target_nodes_ids.append(j)
  edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))
  edge_index_tensor = torch.from_numpy(edge_index)
  return edge_index_tensor

def GetGraphAdjMtrx(squareMtx, thresolds, keep_weights=False): #Apply Thresolds to squareMtx
    graphs = []
    mtxLen = squareMtx.shape[0]
    for thr in thresolds:
        m = np.zeros((mtxLen,mtxLen))#r = []        
        for i in range(0,mtxLen):
            for j in range(0,mtxLen):
                if i == j:# or squareMtx[i,j] > thr:
                    m[i,j] = 1
                elif squareMtx[i,j] > thr:
                  if keep_weights == True:
                    m[i,j] = squareMtx[i,j]
                  else:
                    m[i,j] = 1
        graphs.append(m)#np.array(r))  
    return graphs[0]

from sklearn.preprocessing import StandardScaler

def get_adj_mat(c, th=0, keep_weights=True):
  #print("Creating graph with th: ", th)
  n = c.shape[0]
  a = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      #print("before:", c[i,j])
      if(c[i,j]>th):
        if(keep_weights):
          a[i,j] = c[i,j]
          a[j,i] = c[j,i]
        else:
          a[i,j] = 1
          a[j,i] = 1
      #print("after:", a[i,j])
  return a

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def get_edge_index_weight_tensor(adj):
  num_nodes = adj.shape[0]
  source_nodes_ids, target_nodes_ids, edge_weights = [], [], []
  for i in range(num_nodes):
    for j in range(num_nodes):
      if(adj[i,j]>0):
        source_nodes_ids.append(i)
        target_nodes_ids.append(j)
        edge_weights.append(adj[i,j])
  edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))
  edge_index_tensor = torch.from_numpy(edge_index)
  edge_weights_np = np.asarray(edge_weights, dtype=np.float32)
  edge_weights_tensor = torch.from_numpy(edge_weights_np)
  #print("Index shape: ",edge_index_tensor.shape)
  #print("Weight shape: ",edge_weights_tensor.shape)
  #print(edge_index_tensor)
  #print(edge_weights_tensor)
  return edge_index_tensor, edge_weights_tensor

def normalize_node_attributes(mvts):
  sc = StandardScaler()
  mvts_std = sc.fit_transform(mvts)
  return mvts_std

# RunEpochs get_accuracy trian test acc
def RunEpochs(num_epochs = 1, print_loss_interval = 5): 
  for epoch in range(num_epochs):
    for i in range(num_train):#num_train
      model.zero_grad()

      class_scores = model(train_adjs[i], train_nats[i]) 
      #target = [y_train[i]]
      target = torch.from_numpy(np.array([y_train[i]]))
      target = target.to(device)
      loss = loss_function(class_scores, target)
      loss.backward()
      optimizer.step()
    if(epoch % print_loss_interval == 0):
      print ("epoch n loss:", epoch, loss)

#------------------------------train acc
def get_train_accuracy():
  num_train = X_train.shape[0]
  with torch.no_grad():
    numCorrect = 0
    for i in range(num_train):
      train_class_scores = model(train_adjs[i], train_nats[i])
      class_prediction = torch.argmax(train_class_scores, dim=-1) 
  
      if(class_prediction == y_train[i]): 
        numCorrect = numCorrect + 1
    return numCorrect/num_train


#---------test acc
def get_test_accuracy():
  num_test = X_test.shape[0]
  with torch.no_grad():
    numCorrect = 0
    for i in range(num_test):
      test_class_scores = model(test_adjs[i], test_nats[i]) #(adj_mat_array, node_att_array)
      class_prediction = torch.argmax(test_class_scores, dim=-1) 
      
      if(class_prediction == y_test[i]): 
        numCorrect = numCorrect + 1
    return numCorrect/num_test

def get_accuracy():
  print ("train_accuracy:", get_train_accuracy())
  print ("test_accuracy: ", get_test_accuracy())

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score

def get_accuracy_report_by_running_epochs(epochs, epoch_interval):
  maxAcc=0
  max_classification_report_dict=0
  max_acc_epoch = 0
  num_test = X_test.shape[0]

  for epoch in range(epoch_interval, epochs, epoch_interval):
    print("current epoch: ", epoch)
    RunEpochs(num_epochs = epoch_interval, print_loss_interval = 300)
    
    #get_accuracy()
    with torch.no_grad():
      numCorrect = 0
      predictaedLabel=[]
      for i in range(num_test):
        test_class_scores = model(test_adjs[i], test_nats[i]) #(adj_mat_array, node_att_array)
        class_prediction = torch.argmax(test_class_scores, dim=-1) 
        predictaedLabel.append(class_prediction)
        if(class_prediction == y_test[i]): 
          numCorrect = numCorrect + 1
      acc = numCorrect/num_test
      if acc  > maxAcc: #fgdg=round(acc, 2)
        maxAcc=acc
        max_acc_epoch = epoch
        max_classification_report_dict=metrics.classification_report(y_test, predictaedLabel, digits=3,output_dict=True)

  return maxAcc, max_acc_epoch, max_classification_report_dict

def doClassSpecificCalulcation(Accuracy,trainLebel,classification_report_dict):
  print('\np.mean(Accuracy) :',np.mean(Accuracy))
  print('\np.std(Accuracy) :',np.std(Accuracy))
  print('\n33333333 p.mean np.std(Accuracy) :     ',np.round(np.mean(Accuracy),2),"+-",np.round(np.std(Accuracy),2) )
  for j in [0, 1, 2, 3]:#np.unique(trainLebel ): #. len(...) np.unique(trainLebel):  [0 1 2 3]
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

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
# %matplotlib inline

import seaborn as sns


# Utility function to visualize the outputs of PCA and t-SNE

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(14, 14))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []
    lbls = ["X", "M", "BC", "Q"]
    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, lbls[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


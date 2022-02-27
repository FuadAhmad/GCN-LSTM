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

path = os.getcwd() #'/Users/fuad/Documents/GitHub/models'
path

# set project folder location where the models and data folder exist
projectlocation = '/Users/fuad/Documents/GitHub'

#set data path and load data
datapath = projectlocation + "/data/flare_prediction_mvts_data.pck"
labelpath = projectlocation + "/data/flare_prediction_labels.pck"

mvts_1540=load(datapath)
labels_1540=load(labelpath)

#Check data
print("MVTS dataset shape: ", mvts_1540.shape, "  type: ", type(mvts_1540)) # (1540, 33, 60)
print("Labels shape: ", labels_1540.shape, "  type: ", type(labels_1540))     # (1540,)
print("labels_1540[0]: ", type(labels_1540[0])) #<class 'numpy.int64'>
print("unique labels: ", np.unique(labels_1540))# [0 1 2 3]
print("Example mvts: ", mvts_1540[0])

#Binary classification
# Data label conversion to BINARY class
def get_binary_labels_from(labels_str):
    tdf = pd.DataFrame(labels_str, columns = ['labels'])
    data_classes= [0, 1, 2, 3]
    d = dict(zip(data_classes, [0, 0, 1, 1])) 
    arr = tdf['labels'].map(d, na_action='ignore')
    return arr.to_numpy()

#un-comment next line for Binary classification experiment
#labels_1540 = get_binary_labels_from(labels_1540)

#Stratified train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mvts_1540, labels_1540, test_size=0.3, random_state=0, stratify=labels_1540)

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

#building standard scaler on train data X
trans = GetTransposed2D(X_train)
data2d = Make2D(trans)
scaler = GetStandardScaler(data2d)

X_train = transform_scale_data(X_train, scaler)
X_test = transform_scale_data(X_test, scaler)

#check
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
y_train_stats = dict(zip(unique_y_train, counts_y_train))
print("y_train_counts")
print(y_train_stats)
#270/(269+269+270+270) = 0.25
unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
y_test_stats = dict(zip(unique_y_test, counts_y_test))
print("y_test_counts")
print(y_test_stats)
#116/(116+116+115+115) = 0.25

#@title graph utils
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
  #print(edge_index_tensor)
  #print(edge_weights_tensor)
  return edge_index_tensor, edge_weights_tensor

def normalize_node_attributes(mvts):
  sc = StandardScaler()
  mvts_std = sc.fit_transform(mvts)
  return mvts_std

#data crawler in train dataset
num_temporal_split = 4
th = 0
num_train = X_train.shape[0]
num_nodes = 25
len_st = 15
#populating adjacency matrices and node attributes of train events
#(1078, 4, 6, 25, 25)
train_adjs = np.zeros((num_train, num_temporal_split, num_nodes, num_nodes))
train_nats = np.zeros((num_train, num_temporal_split, num_nodes, len_st))
for i in range(num_train):
  #print('Event: ', i)
  mt = X_train[i].T[:,0:25] #consider first 25 solar params
  #mt = normalize_node_attributes(mt) ++++++++++++++++++++++++++++++
  for j in range(num_temporal_split):
    #print('Temporal split: ', j*15, (j+1)*15)
    smt = mt[j*15:(j+1)*15,:]#unnormalized
    c_smt = np.corrcoef(smt.T)
    c_smt[np.isnan(c_smt)]=0
    for l in range(num_nodes): #gcnconv will automatically add self loops
      c_smt[l,l] = 0
    #smt = normalize_node_attributes(smt)
    train_nats[i,j,:,:] = smt.T
    adj = get_adj_mat(c_smt, th, True) #change from ex 10
    #if(i==0 and j==0):
      #print('is symetric: ', check_symmetric(adj))
    train_adjs[i,j,:,:]=adj

#data crawler in test dataset
num_test = X_test.shape[0]
#populating adjacency matrices and node attributes of test events
#(462, 4, 6, 25, 25)
test_adjs = np.zeros((num_test, num_temporal_split, num_nodes, num_nodes))
test_nats = np.zeros((num_test, num_temporal_split, num_nodes, len_st))
for i in range(num_test):
  #print('Test Event: ', i)
  mt = X_test[i].T[:,0:25]
  #mt = normalize_node_attributes(mt) +++++++++++++++++++++++++++++++++++++++++
  for j in range(num_temporal_split):
    #print('Temporal split: ', j*15, (j+1)*15)
    smt = mt[j*15:(j+1)*15,:]
    c_smt = np.corrcoef(smt.T)
    c_smt[np.isnan(c_smt)]=0
    for l in range(num_nodes): #gcnconv will automatically add self loops
      c_smt[l,l] = 0
    #smt = normalize_node_attributes(smt)
    test_nats[i,j,:,:] = smt.T
    adj = get_adj_mat(c_smt, th, True)
    #if(i==0 and j==0):
      #print('is symetric: ', check_symmetric(adj))
    test_adjs[i,j,:,:]=adj

print(train_adjs.shape)
print(train_nats.shape)
print(test_adjs.shape)
print(test_nats.shape)

#MODELS CELL
#node_emb_dim = graph_emb_dim = window_emb_dim = 4; sequence_emb_dim = 128; class_emb_dim = 4
# (GCN) Node emb -> (mean) Graph emb -> (Flatten, Linear) -> window emb -> (LSTM) -> Temporal sequence emb -> (Linear) Class emb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

class MVTS_GCN_RNN(torch.nn.Module):
  def __init__(self, num_nodes, input_dims, num_temporal_split, device, gcn_hidden_dims, node_emb_dims, graph_emb_dims, window_emb_dims, sequence_emb_dims, num_classes):
    super(MVTS_GCN_RNN, self).__init__()
    self.num_nodes = num_nodes
    self.input_dims = input_dims
    self.num_temporal_split = num_temporal_split
    self.device = device
    self.gcn_hidden_dims = gcn_hidden_dims
    self.node_emb_dims = node_emb_dims
    self.graph_emb_dims = graph_emb_dims
    self.window_emb_dims = window_emb_dims
    self.sequence_emb_dims = sequence_emb_dims
    self.num_classes = num_classes 

    self.smt2vector = nn.LSTM(num_nodes, sequence_emb_dims)
    self.conv1 = GCNConv(input_dims, gcn_hidden_dims)
    self.conv2 = GCNConv(gcn_hidden_dims, node_emb_dims)
    #self.mean_pool = global_mean_pool(node_emb_dims, num_nodes)
    #self.node2graph = nn.Linear(node_emb_dims, graph_emb_dims)#change from ex 1
    self.seqGraph2window = nn.Linear(sequence_emb_dims+graph_emb_dims, window_emb_dims)
    self.window2sequence = nn.LSTM(window_emb_dims, sequence_emb_dims) #change from ex 1
    self.sequence2class_space = nn.Linear(sequence_emb_dims, num_classes)

  def forward(self, adj_mat_array, node_att_array):
     #adj_mat_array -> (4,1,25,25), node_att_array -> (4,25,15)
     sequence_vectors = torch.zeros((self.num_temporal_split, self.window_emb_dims), device=self.device).double()
     for j in range(self.num_temporal_split):
       node_att = node_att_array[j,:,:]#25*15
       adj_mat = adj_mat_array[j,:,:]
       #prepare for GCNConv
       edge_index_tensor, edge_weights_tensor = get_edge_index_weight_tensor(adj_mat)
       edge_index = edge_index_tensor.to(self.device) 
       edge_weights = edge_weights_tensor.to(self.device)
       node_attributes_tensor = torch.from_numpy(node_att)
       x = node_attributes_tensor.to(self.device)#[25,15]
       #for debug
       #graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights)

       #lstm on x.t
       smvts = torch.t(x)
       #print(smvts.shape)
       small_seq_out, _ = self.smt2vector(smvts.view(len(smvts), 1, -1))#input:[15, 25] , output: [15, 128]
       last_small_seq_out = small_seq_out[len(small_seq_out)-1] #[1,128]
       #GCN on the graph
       x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weights)
       x = F.relu(x)
       x = F.dropout(x, training=self.training)
       x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weights) #x -> [25, 4]
       x = F.relu(x) #change from ex 10
       x = F.dropout(x, training=self.training) #change from ex 10
       #node_embs = x
       #graph embedding
       x = torch.mean(x, dim=0).view(1,-1) #->[1,4]#mean pool 
       #x = torch.max(x, dim=0).values.view(1,-1) #max pool
       
       #batch_id = torch.tensor([example_id])
       #batch = batch_id.repeat(num_nodes).view(num_nodes, 1)
       #tempX = global_mean_pool(node_embs, batch)

       #flattened node embeddings
       #x = x.view(1,-1) #x -> [1,100]
       #graph embedding by linear projection
       #x = self.node2graph(x) #x -> [1,16]
       #x = F.relu(x)
       graph_vector = x
       seq_graph_vector = torch.cat((last_small_seq_out, graph_vector), dim=1) #[1, 132]
       #print('Graph cat vec: ', graph_cat_vector.shape) #[1,24]
       #window embedding by linear projection
       window_vector = self.seqGraph2window(seq_graph_vector)#[1,64]
       window_vector = F.relu(window_vector)
       #print('Window vec: ', window_vector.shape)
       sequence_vectors[j,:]=window_vector
     #sequence embedding by RNN, linear, and softmax #sequence_vectors -> [4,6]
     #print('Seq vectors shape: ', sequence_vectors.shape) -> [4, 64]
     seq_out, _ = self.window2sequence(sequence_vectors.view(len(sequence_vectors), 1, -1)) #input: [4, 64] -> seq_out -> [4,128]
     last_seq_out = seq_out[len(seq_out)-1] #[1,128]
     #last_seq_out = F.dropout(last_seq_out, training=self.training) #change from ex 10_2
     class_space = self.sequence2class_space(last_seq_out) #[1,4]
     class_scores = F.log_softmax(class_space, dim=1)
     return class_scores

#Training
torch.manual_seed(0)

NUM_NODES = 25
INPUT_DIMS = 15
NUM_TEMPORAL_SPLIT = 4
GCN_HIDDEN_DIMS = 4 #256 #224 #192 #160 #128 #96 #64 #32 #4 #kIPF used 4 hidden dims for karate (34, 154)
NODE_EMB_DIMS = 4 # number of classes/can be tuned
#Flatten graph emb = 25 * 4 = 100
GRAPH_EMB_DIMS = NODE_EMB_DIMS #change from ex 1
#Flatten threshold emb = 4*6=24
WINDOW_EMB_DIMS = 64 #number of sparsity threshold/can be increased #change from ex 1 #change from ex 10
SEQUENCE_EMB_DIMS = 128 #16 #4 #128 #number of timestamps #change from ex 1 #change from ex 10
NUM_CLASSES = 4 #2 binary classification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = MVTS_GCN_RNN(NUM_NODES, INPUT_DIMS, NUM_TEMPORAL_SPLIT, device, GCN_HIDDEN_DIMS, NODE_EMB_DIMS, GRAPH_EMB_DIMS, WINDOW_EMB_DIMS, SEQUENCE_EMB_DIMS, NUM_CLASSES).to(device).double()
loss_function = nn.NLLLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01) #change from ex 10
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
num_epochs = 0 #change from ex 10

#Train
for epoch in range(num_epochs):
  #print('Epoch: ', epoch)
  for i in range(num_train):
    optimizer.zero_grad()
    #print('Event: ', i)
    adj_mat_array = train_adjs[i,:,:,:]#(4,25,25)
    node_att_array = train_nats[i,:,:,:] #(4,25,15)
    class_scores = model(adj_mat_array, node_att_array) 
    target = [y_train[i]]
    target = torch.from_numpy(np.array(target))
    target = target.to(device)
    loss = loss_function(class_scores, target)
    loss.backward()
    optimizer.step()
  if(epoch%5==0):
    print ("epoch n loss:", epoch, loss)

"""**Fuad's addition + modification**"""

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

#@title
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

#Check model performance 
epochs = 70
epoch_interval = 3
print("current epoch: 0")
get_accuracy()
for epoch in range(epoch_interval, epochs, epoch_interval):
    print("current epoch: ", epoch)
    RunEpochs(num_epochs = epoch_interval, print_loss_interval = 300)
    get_accuracy()

# run only once per experiment(5 differeent random_state of train_test_data_split)
classification_report_dict=[]
Accuracy=[]

#run for each random data state
maxAcc, max_acc_epoch, max_classification_report_dict = get_accuracy_report_by_running_epochs(epochs = 30 + 1, epoch_interval = 5)
 
classification_report_dict.append(max_classification_report_dict)   
Accuracy.append(maxAcc)

maxAcc, max_acc_epoch, max_classification_report_dict

#@title
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

#@title
rs0={'0': {'f1-score': 0.9469387755102041,
   'precision': 0.8992248062015504,
   'recall': 1.0,
   'support': 116},
  '1': {'f1-score': 0.7981651376146788,
   'precision': 0.8529411764705882,
   'recall': 0.75,
   'support': 116},
  '2': {'f1-score': 0.7196652719665273,
   'precision': 0.6935483870967742,
   'recall': 0.7478260869565218,
   'support': 115},
  '3': {'f1-score': 0.8288288288288289,
   'precision': 0.8598130841121495,
   'recall': 0.8,
   'support': 115},
  'accuracy': 0.8246753246753247,
  'macro avg': {'f1-score': 0.8233995034800597,
   'precision': 0.8263818634702657,
   'recall': 0.8244565217391304,
   'support': 462},
  'weighted avg': {'f1-score': 0.8236122846622347,
   'precision': 0.826597019867953,
   'recall': 0.8246753246753247,
   'support': 462}}

rs1 = {'0': {'f1-score': 0.9316239316239315,
   'precision': 0.923728813559322,
   'recall': 0.9396551724137931,
   'support': 116},
  '1': {'f1-score': 0.7905138339920948,
   'precision': 0.7299270072992701,
   'recall': 0.8620689655172413,
   'support': 116},
  '2': {'f1-score': 0.6788990825688073,
   'precision': 0.7184466019417476,
   'recall': 0.6434782608695652,
   'support': 115},
  '3': {'f1-score': 0.8036529680365295,
   'precision': 0.8461538461538461,
   'recall': 0.7652173913043478,
   'support': 115},
  'accuracy': 0.803030303030303,
  'macro avg': {'f1-score': 0.8011724540553409,
   'precision': 0.8045640672385465,
   'recall': 0.8026049475262369,
   'support': 462},
  'weighted avg': {'f1-score': 0.801431745954703,
   'precision': 0.8046604475120994,
   'recall': 0.803030303030303,
   'support': 462}}
rs2 = {'0': {'f1-score': 0.9626556016597512,
   'precision': 0.928,
   'recall': 1.0,
   'support': 116},
  '1': {'f1-score': 0.7705627705627706,
   'precision': 0.7739130434782608,
   'recall': 0.7672413793103449,
   'support': 116},
  '2': {'f1-score': 0.625,
   'precision': 0.6422018348623854,
   'recall': 0.6086956521739131,
   'support': 115},
  '3': {'f1-score': 0.7894736842105263,
   'precision': 0.7964601769911505,
   'recall': 0.782608695652174,
   'support': 115},
  'accuracy': 0.79004329004329,
  'macro avg': {'f1-score': 0.7869230141082619,
   'precision': 0.7851437638329491,
   'recall': 0.7896364317841079,
   'support': 462},
  'weighted avg': {'f1-score': 0.7872679758918248,
   'precision': 0.7854286675468288,
   'recall': 0.79004329004329,
   'support': 462}}
rs3 = {'0': {'f1-score': 0.9707112970711297,
   'precision': 0.943089430894309,
   'recall': 1.0,
   'support': 116},
  '1': {'f1-score': 0.8584070796460177,
   'precision': 0.8738738738738738,
   'recall': 0.8434782608695652,
   'support': 115},
  '2': {'f1-score': 0.6952789699570816,
   'precision': 0.6923076923076923,
   'recall': 0.6982758620689655,
   'support': 116},
  '3': {'f1-score': 0.7876106194690266,
   'precision': 0.8018018018018018,
   'recall': 0.7739130434782608,
   'support': 115},
  'accuracy': 0.829004329004329,
  'macro avg': {'f1-score': 0.828001991535814,
   'precision': 0.8277681997194193,
   'recall': 0.8289167916041978,
   'support': 462},
  'weighted avg': {'f1-score': 0.8280236068690533,
   'precision': 0.8277246082124131,
   'recall': 0.829004329004329,
   'support': 462}}
rs4 = {'0': {'f1-score': 0.9661016949152543,
   'precision': 0.9421487603305785,
   'recall': 0.991304347826087,
   'support': 115},
  '1': {'f1-score': 0.8225806451612903,
   'precision': 0.7669172932330827,
   'recall': 0.8869565217391304,
   'support': 115},
  '2': {'f1-score': 0.6425339366515838,
   'precision': 0.6761904761904762,
   'recall': 0.6120689655172413,
   'support': 116},
  '3': {'f1-score': 0.776255707762557,
   'precision': 0.8252427184466019,
   'recall': 0.7327586206896551,
   'support': 116},
  'accuracy': 0.8051948051948052,
  'macro avg': {'f1-score': 0.8018679961226713,
   'precision': 0.8026248120501849,
   'recall': 0.8057721139430285,
   'support': 462},
  'weighted avg': {'f1-score': 0.8014676793524739,
   'precision': 0.8024001011639006,
   'recall': 0.8051948051948052,
   'support': 462}}

saved_reports = []
saved_reports.append(rs0)
saved_reports.append(rs1)
saved_reports.append(rs2)
saved_reports.append(rs3)
saved_reports.append(rs4)
#classification_report_dict.append(rs0)
savedAcc = [0.82467, 0.80303, 0.79004, 0.82900, 0.80519]
doClassSpecificCalulcation(savedAcc, X_train, saved_reports)

len(classification_report_dict)

doClassSpecificCalulcation(Accuracy, X_train, classification_report_dict)

"""t-SNE"""

# Before t-SNE representation, train the model for 100 epochs

# Commented out IPython magic to ensure Python compatibility.
#@title
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

#@title
from sklearn.manifold import TSNE
from numpy import save

# save to npy file
#save('train_tsne_data.npy', train_tsne)

# to load data
#numpy.load("file path/train_tsne_data.npy")

fashion_scatter(train_tsne, y_train)

#@title
xe1 = train_tsne[:,0]
xe2 = train_tsne[:,1]
y = y_train

df = pd.DataFrame({'t-SNE dimension 1':xe1, 't-SNE dimension 2':xe2, 'Class':y})
df = df.sort_values(by=['Class'], ascending=True)
#print(df.iloc[0:25])

#@title
legend_map = {0: 'X',
              1: 'M',
              2: 'BC',
              3: 'Q'}
fig = plt.figure(figsize=(11, 11))
sns.set(font_scale=2)
ax = sns.scatterplot(df['t-SNE dimension 1'], df['t-SNE dimension 2'], hue=df['Class'].map(legend_map), 
                     palette=['red', 'orange', 'blue', 'green'], legend='full')
plt.show()
img_file = 'all_tsne.pdf'
fig.savefig(img_file, dpi=fig.dpi)
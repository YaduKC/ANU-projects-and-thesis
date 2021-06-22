import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from itertools import combinations
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
from math import pi

# Reading in anger data set and preprocessing it
df = pd.read_csv('Anger.csv')
# Encode output label
df = df.replace('Genuine',0)
df = df.replace('Posed',1)
# Extract the input and output features
x = df.iloc[:,2:8].values
y = df.iloc[:,8].values
# Split data set into train and test in the ratio 80/20
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# Normalise the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.fit_transform(X_test)
# Convert training and testing data into tensors
X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
y_test =torch.tensor(y_test, dtype=torch.float)

# Dataset class for custom train dataloader
class Train_Data(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

train_data = Train_Data(X_train,y_train)   

# Dataset class for custom test dataloader
class Test_Data(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
test_data = Test_Data(X_test)

# Model parameters
Batch_Size = 10
Learning_rate = 0.001
Epochs = 200
input_neurons = 6
hidden_neurons = 24
output_neurons = 1
criterion = nn.BCEWithLogitsLoss()

train_loader = DataLoader(dataset=train_data, batch_size=Batch_Size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

# Simple feedforward neural net structure with one hidden layer
class classifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(classifier, self).__init__()
        # Number of input features is 6.
        self.layer_1 = nn.Linear(n_input, n_hidden) 
        self.layer_out = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(n_hidden)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x_out = self.bn1(x)
        x = self.layer_out(x_out)
        
        return x, x_out

# function to calculate accuracy
def accuracy(y_pred, y_test):
    y_p = torch.round(torch.sigmoid(y_pred))# apply sigmoid to output and round it to get class label
    l = (y_p == y_test).sum().float()
    acc = l/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

net = classifier(input_neurons, hidden_neurons, output_neurons)
net.train()
optimizer = optim.Adam(net.parameters(), lr=Learning_rate)
# Train the network
print("Model parameters : ")
print()
print("Batch size = ", Batch_Size)
print("Learning rate = ", Learning_rate)
print("Epochs = ", Epochs)
print("Hidden neurons = ", hidden_neurons)
print()
print("Model architecture : ")
print()
print(net)
print()
print("Training model............................")
print()
for e in range(1, Epochs+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred,h = net(X_batch)       
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = accuracy(y_pred, y_batch.unsqueeze(1))      
        loss.backward()
        optimizer.step()       
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    # Print progress after every 10 epochs
    if e%10 == 0:
        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

# Evaluate the original model
y_pred_list = []
net.eval()
with torch.no_grad():
    for X_batch in test_loader:
        y_test_pred,h = net(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
print()
print("Classification report for original model")
print()
print(classification_report(y_test, y_pred_list))

# Calculate similar and complementary pairs
y_pred_t,h = net(X_train)
# Extract pattern vector and normalize it
pattern_vec = minmax_scale(h.detach().numpy(),feature_range=(-0.5,0.5))
comp = [] # Stores complementary pairs
sim = [] # Stores similar pairs

for i in list(combinations(np.arange(0,h.shape[1]),2)): # Loop through all the possible vector combinations
    u = pattern_vec[:,i[0]]
    v = pattern_vec[:,i[1]]
    c = dot(u,v)/norm(u)/norm(v)
    angle = (arccos(clip(c, -1, 1))*180)/pi
    if angle < 15: 
        sim.append(i)
    if angle > 165:
        comp.append(i)
        
comp_u = np.unique(comp) # Keep only unique values in the complementary pair list
# Items in complementary pair list signify those neurons that needs to be removed. Removing these neurons means that
# some of the pairs in the similar pair list can also be removed. This is done below.
pop_list = []
k = 0
for i in comp_u:
    for j in sim:
        if j[0] == i or j[1] == i:
            pop_list.append(k)
        k += 1
    k = 0

sim_u = []
c =0
for i in np.unique(pop_list):
    sim_u.append(sim[i])
sim_u = list(set(sim)^set(sim_u))
# All unique similar and complementary pairs
#print(comp_u)
#print(sim_u)

# Setting all input and output weights of calculated neurons to zero 
w_1 = net.layer_1.weight.data
w_2 = net.layer_out.weight.data
w_1_c =(net.layer_1.weight.data).clone()
w_2_c = (net.layer_out.weight.data).clone()

# set complementary neuron weights to zero
for i in comp_u:
    w_1[i,:] = 0
    w_2[:,i] = 0
    
# dealing with similar pairs
for i in sim_u:
    w_1[i[0],:] = w_1[i[0],:] + w_1_c[i[1],:]
    w_2[:,i[0]] = w_2[:,i[0]] + w_2_c[:,i[1]]
for j in sim_u:
    w_2[:,j[1]] = 0
    w_1[j[1],:] = 0

# Testing updated model
y_pred_list = []
net.eval()
with torch.no_grad():
    for X_batch in test_loader:
        y_test_pred,h = net(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
print()
print("Classification report for reduced model")
print()
print(classification_report(y_test, y_pred_list))
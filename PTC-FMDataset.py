import os.path as osp
import torch, torch_geometric,random
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch_geometric.data import Dataset
from torch_geometric.data import Data, DataLoader
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle


nb_graph = 349
data_list = [0]
min_values, max_values = [0 for _ in range(350)], [0 for _ in range(350)]
min_values[1], max_values[1], max_values[349], idx = 1, 2, 4925, 1

f = open("PTC-FM.graph_idx", "r")
for row in range(1,4926):
    n = int(f.readline())
    if n != idx:
        max_values[idx] = row - 1
        idx += 1
        min_values[idx] = row



edgeList_minValues, edgeList_maxValues = [0 for _ in range(350)], [0 for _ in range(350)]
f = open("PTC-FM.edges","r")
graph_ind = 1
for row in range(1,54*187+13):
    v1,v2 = f.readline().split(sep=",")
    v1,v2 = int(v1), int(v2)
    if v1>max_values[graph_ind]:
        edgeList_maxValues[graph_ind] = row - 1
        edgeList_minValues[graph_ind + 1] = row
        graph_ind += 1

edgeList_minValues[1], edgeList_maxValues[349] = 1, 54*187+12


f_nodeLabels = open("PTC-FM.node_labels","r")
f_edge = open("PTC-FM.edges","r")
f_edgeLabels = open("PTC-FM.link_labels","r")
f_graphLabels = open("PTC-FM.graph_labels","r")

graph_ind, edge_ind, node_ind = 1, 1, 1



while graph_ind < 350:
    nb_node = max_values[graph_ind] - min_values[graph_ind] + 1
    x = np.zeros(shape=(nb_node, 18))
    for _ in range(nb_node):
        lbl = int(f_nodeLabels.readline())
        x[_,lbl] = 1
    nb_edge = edgeList_maxValues[graph_ind] - edgeList_minValues[graph_ind] + 1
    edge1, edge2 = np.array([0 for _ in range(nb_edge)]), np.array([0 for _ in range(nb_edge)])
    edge_matrix = np.zeros(shape=(nb_edge, 4))
    for _ in range(nb_edge):
        v1, v2 = f_edge.readline().split(sep=",")
        v1, v2 = int(v1), int(v2)
        edge1[_], edge2[_] = v1, v2
        lbl = int(f_edgeLabels.readline())
        edge_matrix[_,lbl] = 1

    edge1, edge2 = edge1-min_values[graph_ind], edge2-min_values[graph_ind]
    out_y = 1 if int(f_graphLabels.readline())==1 else 0
    y = np.array([out_y])
    data_list.append(Data(x=torch.tensor(data=x, dtype=torch.float), edge_attr=torch.tensor(data=edge_matrix, dtype=torch.float),
                          edge_index=torch.tensor(data=[edge1,edge2], dtype=torch.long),
                          y = torch.tensor(data=y, dtype=torch.float)))
    graph_ind += 1
del data_list[0]

loader = DataLoader(data_list, batch_size = 32, shuffle=True)



class NeuralNetwork(torch.nn.Module):
    def __init__(self):

        super(NeuralNetwork,self).__init__()
        self.nnconv1 = torch_geometric.nn.NNConv(in_channels=18, out_channels=10, nn=torch.nn.Sequential(nn.Linear(in_features=4,out_features=180)))
        self.nnconv2 = torch_geometric.nn.NNConv(in_channels=10, out_channels=7, nn=torch.nn.Sequential(nn.Linear(in_features=4, out_features=70)))
        self.w1 = torch.nn.Linear(in_features=7,out_features=20,bias=True)
        self.w2 = torch.nn.Linear(in_features=20,out_features=10,bias=True)
        self.w3 = torch.nn.Linear(in_features=10, out_features=1, bias=True)

    def forward(self, data):

        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr
        x = self.nnconv1(x,edge_index,edge_attr)
        x = self.nnconv2(x,edge_index,edge_attr)
        nb_nodes = int(x.shape[0])
        cluster = torch.tensor(data=np.zeros(shape=(nb_nodes,)))
        btc = torch.tensor(data=np.zeros(shape=(nb_nodes,)))
        x = torch_geometric.nn.avg_pool_x(cluster=cluster,x = x,batch=btc)[0]
        x = F.relu(self.w1(x))
        x = F.relu(self.w2(x))
        x = self.w3(x)
        x = torch.sigmoid(x)
        return x



Network = NeuralNetwork()
optimizer = optim.SGD(params=Network.parameters(),lr=0.0005,momentum=0.9)
criterion = nn.BCELoss()
nb_epoch = 1500
total_lost = 0

data_list_train = data_list[:299]
data_list_test = data_list[299:]

for _ in range(nb_epoch):
    ls = 0
    data_list_train = shuffle(data_list_train)
    for inp in data_list_train:
        optimizer.zero_grad()
        out = Network(inp)
        out = out.reshape(shape=(1,))
        loss = criterion(out,inp.y)
        ls += loss.data
        loss.backward()
        optimizer.step()
    print(ls)




wrong_answers = 0

for v in data_list_test:
    o = Network(v).data
    o = 1 if o>(1/2) else 0
    if o != v.y.data:
        wrong_answers += 1

print(100*wrong_answers/len(data_list_test))



from torch_geometric.data import Data
import numpy as np
import torch
import tqdm
from scipy.spatial.distance import cosine



class GenerateGraph:
    
    def __init__(self,node_features,node_num):
      self.node_features = node_features
      self.node_num = node_num

    def matrix_of_distance(self):
        self.node_features = self.node_features.cpu()
        distance_matrix = np.zeros([self.node_num,self.node_num])
        for i in tqdm.trange(self.node_num):
            for j in range(self.node_num):
                dis = cosine(self.node_features[i],self.node_features[j])
                distance_matrix[i,j] = dis
        return distance_matrix
    
    def find_edges(self,matrix,th=0.2):
        indicies = np.zeros([2,1])
        for i in range(self.node_num):
            for j in range(self.node_num):
                if matrix[i,j] < th and matrix[i,j] != 0:
                    temp = np.array([i,j]).reshape(2,1)
                    indicies = np.concatenate([indicies,temp],axis=1)
    
        return indicies[:,1:]

    def PytochGraph(self,indices,label):
      indices = torch.tensor(indices, dtype=torch.long)
      self.node_features = torch.tensor(self.node_features,dtype=torch.float)
      # z = torch.tensor([[-1], [0], [1]], dtype=torch.float)
      data = Data(x=self.node_features, edge_index=indices,y=torch.tensor([label]))
      return data
       
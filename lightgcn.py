from scipy.sparse import coo_matrix
from scipy.sparse import diags 
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch
from logger import Logger

#torch.autograd.set_detect_anomaly(True)    

class Train_data(Dataset):
    def __init__(self, data):
        super(Train_data,self).__init__()
        self.S = torch.tensor(data).float()
    
    def __len__(self):
        return len(self.S)
    
    def __getitem__(self, index):
        return self.S[index]

class LightGCN(nn.Module):
    def __init__(self,fname, lamda = 1e-4, lr = 3e-4, latent_dim = 64, device = torch.device('cpu')):
        super(LightGCN, self).__init__()

        self.device = device
         
        self.mat = self.load_data(fname).to(self.device)
        
        #print("graph", self.mat)
        
        self.logger = Logger()

        self.lamda = lamda
        self.user_emb = nn.Embedding(self.n_users, latent_dim).double()
        self.item_emb = nn.Embedding(self.n_items, latent_dim).double()
        
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        
        self.optimizer = Adam(self.parameters(), lr = lr)
        
    def forward(self, stages = 3):
        emb = torch.cat([self.user_emb.weight, self.item_emb.weight], axis = 0)    
        emb_list = [emb]
        
        for i in range(stages):
            #print("emb norm", torch.norm(emb).item())
            emb = torch.sparse.mm(self.mat,emb)
            emb_list.append(emb)
        
        return torch.mean(torch.stack(emb_list, dim = 1), dim = 1), emb_list[0]
    
    def bpr_loss(self, S, emb, init_emb):
        
        S = np.array(S).astype('int')
        
        all_user_emb, all_item_emb = torch.split(emb,[self.n_users,self.n_items])
        all_user_emb0, all_item_emb0 = torch.split(init_emb,[self.n_users,self.n_items])
        
        pos_emb = all_item_emb[S[:,1]]
        neg_emb = all_item_emb[S[:,2]]      
        user_emb = all_user_emb[S[:,0]]

        #print("pos norm",torch.norm(pos_emb).item())
        #print("neg norm",torch.norm(neg_emb).item())
        #print("user norm",torch.norm(user_emb).item())

        pos_emb0 = all_item_emb0[S[:,1]]
        neg_emb0 = all_item_emb0[S[:,2]]      
        user_emb0 = all_user_emb0[S[:,0]]
    
        loss = (F.softplus(torch.sum(user_emb*neg_emb, dim = 1) - torch.sum(user_emb*pos_emb, dim =1))).mean()             
        
        #print("loss1",loss.item())

        loss += self.lamda*(torch.norm(pos_emb0)**2 + torch.norm(neg_emb0)**2 +  torch.norm(user_emb0)**2)/float(len(pos_emb))
        
        #print("loss total",loss.item())

        return loss
    
    def train_model(self, n_iters = 100, stages = 3):
        
        for i in range(n_iters):
            print("Iteration number", i,flush = True)
            S = Train_data(self.sample_interaction())
            train_loader = DataLoader(S, batch_size = 2048, shuffle = True)
            for batch, sample in enumerate(train_loader):
                emb, init_emb = self(stages)
                loss = self.bpr_loss(sample, emb, init_emb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.logger.add_scalar("BPR loss",loss.item())
                #print("bpr loss", loss)
            self.logger.add_scalar("NDCG score",self.evaluate(20))
    
    def sample_interaction(self):
        
        users = np.random.randint(0, self.n_users, self.n_interaction)
        S = []
        
        for user in users:
            user_item = self.train[user]
            pos_item = user_item[np.random.randint(0, len(user_item))]
            while True:
              neg_item = np.random.randint(0, self.n_items)
              if(neg_item in user_item):
                  continue
              else:
                  break
            S.append([user, pos_item, neg_item])
        
        return np.array(S).astype('int')
        

    def load_data(self,fname):

        file = open(fname,'r')
        train = {}
        users = set()
        items = set()
        
        count = 0
        
        self.n_interaction = 0
        
        while True:
            line = file.readline()[:-1]
            if(line == ''):
                break
            temp = line.split(' ')
            temp = list(map(lambda x: int(x),temp))
            
            count += (len(temp) -1)
            
            users.add(temp[0])
            
            self.n_interaction += len(temp[1:])
            
            items = items.union(set(temp[1:]))
            
            if(train.get(temp[0]) is None):
                train[temp[0]] = []
                train[temp[0]].extend(temp[1:])
            else:
                train[temp[0]].extend(temp[1:])
        
        self.train = train
        self.users = users
        self.items = items
        
        row_arr = np.zeros(2*count, dtype = np.int32)
        col_arr = np.zeros(2*count, dtype = np.int32)
        data = np.ones(2*count, dtype = np.int32) 
        
        self.n_users = len(users)
        self.n_items = len(items)
        
        count = 0
        for key in train.keys():
            for value in train[key]:
                row_arr[count] = int(key)
                col_arr[count] = len(users)+int(value)
                count+=1
        
        for key in train.keys():
            for value in train[key]:
                row_arr[count] = len(users) + int(value)
                col_arr[count] = int(key)
                count+=1
        
        mat = coo_matrix((data,(row_arr,col_arr)))
        d_mat = mat.sum(axis = 1)
        d_mat = np.sqrt(d_mat)
        d_mat = np.array(d_mat)
        d_mat = 1/(d_mat.reshape(-1))
        d_mat = diags(d_mat)
        d_mat = d_mat.tocoo()
        final = (d_mat@mat@d_mat).tocoo()
        
        rows = torch.tensor(final.row)
        cols = torch.tensor(final.col)
        index = torch.cat([rows.reshape(1,-1), cols.reshape(1,-1)], axis = 0)
        return torch.sparse_coo_tensor(index,torch.tensor(final.data))
    
    def load_test_data(self, test_name):
        
        file = open(test_name,'r')
        test = {}
                
        while True:
            line = file.readline()[:-1]
            if(line == ''):
                break
            temp = line.split(' ')
            temp = list(map(lambda x: int(x),temp))
                      
            if(test.get(temp[0]) is None):
                test[temp[0]] = []
                test[temp[0]].extend(temp[1:])
            else:
                test[temp[0]].extend(temp[1:])
        
        self.test = test
    
    def evaluate(self, k):
        eval_score = 0
        users = Train_data(np.arange(self.n_users))
        user_loader = DataLoader(users, batch_size=2048)
        for batch, sample in enumerate(user_loader):
            user_emb = self.user_emb(sample.long().to(self.device))
            score = user_emb @ self.item_emb.weight.T
            temp , top_items = torch.topk(score,k,dim = 1)
            pred_matrix = []
            truth_matrix = np.zeros((len(sample),k))
            for i in sample:
                i = i.long().item()
                pred_matrix.append(list(map(lambda x: int(x in self.test[i]),top_items[i%2048])))
                length = min(k,len(self.test[i]))
                truth_matrix[i%2048,:length] = 1
            
            pred_matrix = np.array(pred_matrix)
            
            idcg = np.sum(truth_matrix*(1/np.log(np.arange(2,k+2))), axis = 1)
            
            dcg = np.sum(pred_matrix*(1/np.log(np.arange(2,k+2))), axis = 1)
            
            idcg[idcg == 0] = 1

            ndcg = dcg/idcg
            
            eval_score += np.sum(ndcg)
        
        return eval_score/(self.n_users)
            
            
            
            
        
        

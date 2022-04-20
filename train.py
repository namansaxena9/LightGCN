from lightgcn import LightGCN

import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = LightGCN('yelp2018/train.txt', lr = 1e-3,device = 'cuda')

model.to(device)

model.load_test_data('yelp2018/test.txt')

model.train_model()

model.evaluate(20)



import torch
import torch.nn as nn
import random
from torch.onnx.symbolic_opset9 import linalg_norm
import math
from utils import sparse_dropout, spmm,get_weight
import numpy as np
def random_sample_alternative(x, y):
    if y > x + 1:
        raise ValueError("y cannot be greater than the range size (x + 1)")
    nums = list(range(x + 1))
    random.shuffle(nums)  # 随机打乱列表
    return nums[:y]       # 返回前 y 个数
class LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, lambda_3,dropout, batch_user, device):
        super(LightGCL,self).__init__()

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))

        self.train_csr = train_csr
        self.adj_norm = adj_norm

        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

    def preprocessing(self,mashup_num, api_num):
        Alist = []
        Mlist = []
        rows, cols = mashup_num, mashup_num
        Mmap = [[0] * cols for _ in range(rows)]
        rows, cols = api_num, api_num
        Amap = [[0] * cols for _ in range(rows)]
        # Amap = {}
        # Mmap = {}
        Amax = 1
        Mmax = 1

        # 预处理不同api在同一个mashup中出现的次数 Amap[i][j]代表第i个api和第j个api在同一个mashup中出现的次数
        with open("data/train.txt", "r") as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                line = [int(i) for i in line]
                items = line[1:]
                for i in range(len(items)):
                    for j in range(len(items)):
                        if items[i] == items[j]:
                            continue
                        else:
                            Amap[items[i]][items[j]] += 1
                            Amax = max(Amax, Amap[items[i]][items[j]])

        # 预处理不同mashup中包含相同api的数量,Mmap[mashup1][mashup2]代表mashup1和mashup2中包含相同api的数量
        with open("data/train.txt", "r") as f:
            lines = f.readlines()
            for line1 in lines:
                line1 = line1.strip().split(' ')
                line1 = [int(i) for i in line1]
                mashup1 = line1[0]
                items1 = line1[1:]
                for line2 in lines:
                    line2 = line2.strip().split(' ')
                    line2 = [int(i) for i in line2]
                    mashup2 = line2[0]
                    items2 = line2[1:]
                    if mashup1==mashup2:
                        continue
                    for item in items2:
                        if item in items1:
                            Mmap[mashup1][mashup2] += 1
                            Mmax = max(Mmax, Mmap[mashup1][mashup2])

        for i in range(mashup_num):
            Newlist = []
            for j in range(mashup_num):
                if(Mmap[i][j]>0):
                    Newlist.append(j)
            Mlist.append(Newlist)

        for i in range(api_num):
            Newlist = []
            for j in range(api_num):
                if(Amap[i][j]>0):
                    Newlist.append(j)
            Alist.append(Newlist)

        self.Amap = Amap
        self.Mmap = Mmap
        self.Amax = Amax
        self.Mmax = Mmax
        self.Alist = Alist
        self.Mlist = Mlist
        self.mashup_num = mashup_num
        self.api_num = api_num
        # for i in range(len(Alist)):
        #     print(len(Alist[i]))

        print(Alist[0],Alist[1],Alist[2])
        print(Mlist[0],Mlist[1],Mlist[2])

    def forward(self, uids, iids, pos, neg, test=False):
        if test==True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            for layer in range(1,self.l+1):
                # GNN propagation
                # self.G_u 和 self.G_i：聚合所有层次的 SVD 增强特征。
                # self.E_u 和 self.E_i：聚合所有层次的 GNN 传播特征。
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

                # svd_adj propagation
                vt_ei = self.vt @ self.E_i_list[layer-1]
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]

            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)

            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # cl loss
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i
            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
            loss_s = -pos_score + neg_score

            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)

            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2
            # print(loss_r,loss_reg,loss_s)
            # cacl loss
            Acacl = 0
            Mcacl = 0

            # for i in range(len(uids)):
            #     Mvis[uids[i]] = 1
            # for i in range(len(iids)):
            #     Avis[iids[i]] = 1
            Mashup_list = random_sample_alternative(self.mashup_num-1,100)
            Api_list = random_sample_alternative(self.api_num-1,100)
            Mvis = [0 for _ in range(self.mashup_num)]
            Avis = [0 for _ in range(self.api_num)]
            for i in Mashup_list:
                Mvis[i] = 1
            for i in Api_list:
                Avis[i] = 1
            for i in Mashup_list:
                for j in range(len(self.Mlist[i])):
                    uid2 = self.Mlist[i][j]
                    if Mvis[uid2] == 0:
                        continue
                    uid1 = i
                    Num = self.Mmap[uid1][uid2]
                    Weight = 1/(1+math.exp((-5)*(Num/self.Mmax)))
                    # Mcacl += 1/(1+math.exp(self.Mmax-5-Num)) * torch.norm(self.E_u[uid1]-self.E_u[uid2],p=2)
                    temp  = torch.norm(self.E_u[uid1] - self.E_u[uid2],p=2)
                    Mcacl += Weight * temp
                    #     # total+= temp
                    #     # times1 += Weight
                    #     # times2 += 1/(1+math.exp(self.Mmax-5-Num))

            for i in Api_list:
                for j in range(len(self.Alist[i])):
                    iid2 = self.Alist[i][j]
                    if Avis[iid2] == 0:
                        continue
                    iid1 = i
                    Num = self.Amap[iid1][iid2]
                    Weight = 1 / (1 + math.exp((-5) * (Num / self.Amax)))
                    # Acacl += 1/(1+math.exp(self.Amax-5-Num)) *torch.norm(self.E_i[iid1]-self.E_i[iid2],p=2)
                    temp  = torch.norm(self.E_i[iid1] - self.E_i[iid2], p=2)
                    Acacl += Weight * temp
                    # total += temp
                    # times1 += Weight
                    # times2 += 1 / (1 + math.exp(self.Mmax - 5 - Num))

            cacl_loss = self.lambda_3 * (Acacl + Mcacl)
            # print(times1,times2,total)
            # # print(times)
            # # print(cacl_loss)
            # # total loss
            # print(cacl_loss)
            loss = loss_r + self.lambda_1 * loss_s + loss_reg + cacl_loss
            # print("loss:",loss)
            #print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            return loss, loss_r, self.lambda_1 * loss_s


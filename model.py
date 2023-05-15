from cmath import cos
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy.special import lambertw
import random
import scipy.sparse as sp
import faiss


class MF(nn.Module):
    def __init__(self, args, data):
        super(MF, self).__init__()
        self.name = args.modeltype
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.decay = args.regs
        self.device = torch.device(args.cuda)
        self.saveID = args.saveID
        self.train_norm = args.train_norm
        self.pred_norm = args.pred_norm

        self.train_user_list = data.train_user_list
        self.valid_user_list = data.valid_user_list
        # = torch.tensor(data.population_list).cuda(self.device)
        self.user_pop = torch.tensor(data.user_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.item_pop = torch.tensor(data.item_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.user_pop_max = data.user_pop_max
        self.item_pop_max = data.item_pop_max        

        self.embed_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embed_item = nn.Embedding(self.n_items, self.emb_dim)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

    # Prediction function used when evaluation
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = all_items[torch.tensor(items).cuda(self.device)]
        if(self.modeltype != 'LGN'):
            users = F.normalize(users, dim = -1)
            items = F.normalize(items, dim = -1)
        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) # 返回一个u*i的矩阵就行 290*300 for coat 返回值需要是u*i就行

        return rate_batch.cpu().detach().numpy()

class LGN(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph()
        self.n_layers = args.n_layers
        self.modeltype = args.modeltype

    def compute(self):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        if(self.train_norm == True):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)
        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = all_items[torch.tensor(items).cuda(self.device)]
        if(self.pred_norm == True):
            users = F.normalize(users, dim = -1)
            items = F.normalize(items, dim = -1)

        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) # 返回一个u*i的矩阵就行 290*300 for coat 返回值需要是u*i就行

        return rate_batch.cpu().detach().numpy()

class XSimGCL(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph()
        self.n_layers = args.n_layers
        self.cl_rate = args.lambda_cl
        self.temp = args.temp_cl
        self.eps = args.eps_XSimGCL
        self.layer_cl = args.layer_cl

    def compute(self, perturbed=False):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = []
        emb_cl = all_emb
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            if perturbed:
                random_noise = torch.rand_like(all_emb).cuda() # add noise
                all_emb += torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.eps
            embs.append(all_emb)
            if layer==self.layer_cl-1:
                emb_cl = all_emb
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.n_users, self.n_items])
        users_cl, items_cl = torch.split(emb_cl, [self.n_users, self.n_items]) # view of noise

        if perturbed:
            return users, items, users_cl, items_cl
        return users, items
    
    def InfoNCE(self, view1, view2, temperature, b_cos = True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score+10e-6)
        return torch.mean(cl_loss)
    
    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        # 算的一个batch中的
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def forward(self, users, pos_items, neg_items):
        all_users, all_items, all_users_cl, all_items_cl = self.compute(perturbed=True)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        # contrastive loss
        cl_loss = self.cl_rate * self.cal_cl_loss([users,pos_items], all_users, all_users_cl, all_items, all_items_cl)

        # main loss
        # use cosine similarity to calculate the scores
        if(self.train_norm == True):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-6)
        mf_loss = torch.negative(torch.mean(maxi))

        # regularizer loss
        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, cl_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = all_items[torch.tensor(items).cuda(self.device)]
        
        if(self.pred_norm == True):
            users = F.normalize(users, dim = -1)
            items = F.normalize(items, dim = -1)
        
        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) # 返回一个u*i的矩阵就行 290*300 for coat 返回值需要是u*i就行

        return rate_batch.cpu().detach().numpy()


class SGL(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph()
        self.n_layers = args.n_layers
        self.cl_rate = args.lambda_cl
        self.temp = args.temp_cl

    def compute(self, perturbed_adj=None):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]


        for layer in range(self.n_layers):
            if perturbed_adj is not None:
                all_emb = torch.sparse.mm(perturbed_adj, all_emb)
            else:
                all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items
    
    def get_enhanced_adj(self, ui_mat, drop_rate):
        adj_shape = ui_mat.get_shape()
        edge_count = ui_mat.count_nonzero()
        row_idx, col_idx = ui_mat.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - drop_rate)))
        user_np = np.array(row_idx)[keep_idx]
        item_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)

        adj_shape = dropped_adj.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = dropped_adj.nonzero()
        ratings_keep = dropped_adj.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T

        shape = tmp_adj.get_shape()
        rowsum = np.array(tmp_adj.sum(1))

        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(tmp_adj)
        norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)

        coo = norm_adj_mat.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        enhanced_adj_mat = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

        return enhanced_adj_mat.coalesce().cuda(self.device)

    def InfoNCE(self, view1, view2, temperature, b_cos = True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score+10e-6)
        return torch.mean(cl_loss)

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.compute(perturbed_mat1)
        user_view_2, item_view_2 = self.compute(perturbed_mat2)
        # view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        # view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)
        user_cl_loss = self.InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = self.InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss
        # return self.InfoNCE(view1,view2,self.temp)

    def forward(self, users, pos_items, neg_items, dropped_adj1, dropped_adj2):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        # contrastive loss
        cl_loss = self.cl_rate * self.cal_cl_loss([users,pos_items], dropped_adj1, dropped_adj2)

        # main loss
        # use cosine similarity as prediction score
        if(self.train_norm == True):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-6)
        mf_loss = torch.negative(torch.mean(maxi))

        # regularizer loss
        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, cl_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = all_items[torch.tensor(items).cuda(self.device)]

        if(self.pred_norm == True):
            users = F.normalize(users, dim = -1)
            items = F.normalize(items, dim = -1)

        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) # 返回一个u*i的矩阵就行 290*300 for coat 返回值需要是u*i就行

        return rate_batch.cpu().detach().numpy()


class NCL(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.emb_size = args.embed_size
        self.Graph = data.getSparseGraph()
        self.n_layers = args.n_layers
        self.ssl_reg = args.lambda_cl
        self.ssl_temp = args.temp_cl
        self.proto_reg = args.proto_reg
        self.alpha = args.ncl_alpha
        self.k = args.num_clusters
        self.ncl_start_epoch = args.ncl_start_epoch


    def compute(self):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        stack_embs = torch.stack(embs, dim=1)
        light_out = torch.mean(stack_embs, dim=1)

        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items, embs

    def InfoNCE(self, view1, view2, temperature, b_cos = True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score+10e-6)
        return torch.mean(cl_loss)

    
    def e_step(self):
        user_embeddings = self.embed_user.weight.detach().cpu().numpy()
        item_embeddings = self.embed_item.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x        """
        kmeans = faiss.Kmeans(d=self.emb_size, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, 1)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).cuda()
        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster

    def ProtoNCE_loss(self, initial_emb, user_idx, item_idx):
        user_emb, item_emb = torch.split(initial_emb, [self.n_users, self.n_items])
        user2cluster = self.user_2cluster[user_idx]
        user2centroids = self.user_centroids[user2cluster]
        proto_nce_loss_user = self.InfoNCE(user_emb[user_idx],user2centroids,self.ssl_temp) * self.batch_size
        item2cluster = self.item_2cluster[item_idx]
        item2centroids = self.item_centroids[item2cluster]
        proto_nce_loss_item = self.InfoNCE(item_emb[item_idx],item2centroids,self.ssl_temp) * self.batch_size
        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, context_emb, initial_emb, user, item):
        context_user_emb_all, context_item_emb_all = torch.split(context_emb, [self.n_users, self.n_items])
        initial_user_emb_all, initial_item_emb_all = torch.split(initial_emb, [self.n_users, self.n_items])
        context_user_emb = context_user_emb_all[user]
        initial_user_emb = initial_user_emb_all[user]
        norm_user_emb1 = F.normalize(context_user_emb)
        norm_user_emb2 = F.normalize(initial_user_emb)
        norm_all_user_emb = F.normalize(initial_user_emb_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        context_item_emb = context_item_emb_all[item]
        initial_item_emb = initial_item_emb_all[item]
        norm_item_emb1 = F.normalize(context_item_emb)
        norm_item_emb2 = F.normalize(initial_item_emb)
        norm_all_item_emb = F.normalize(initial_item_emb_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def forward(self, users, pos_items, neg_items, epoch):
        all_users, all_items, emb_list = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        # contrastive loss
        initial_emb = emb_list[0]
        context_emb = emb_list[2]
        ssl_loss = self.ssl_layer_loss(context_emb,initial_emb,users,pos_items)

        # main loss
        # use cosine similarity as prediction score
        if(self.train_norm == True):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-6)
        mf_loss = torch.negative(torch.mean(maxi))

        # regularizer loss
        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        # proto loss
        if epoch >= self.ncl_start_epoch:
            proto_loss = self.ProtoNCE_loss(initial_emb, users, pos_items)
            return mf_loss, ssl_loss, proto_loss, reg_loss
        else:
            return mf_loss, ssl_loss, torch.tensor(0.0), reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items, _ = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = all_items[torch.tensor(items).cuda(self.device)]

        if(self.pred_norm == True):
            users = F.normalize(users, dim = -1)
            items = F.normalize(items, dim = -1)

        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) # 返回一个u*i的矩阵就行 290*300 for coat 返回值需要是u*i就行

        return rate_batch.cpu().detach().numpy()

class IPS(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items, pos_weights):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.mul(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10), pos_weights)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

class CausE(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.cf_pen = args.cf_pen
        self.embed_item_ctrl = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.xavier_normal_(self.embed_item_ctrl.weight)
    
    
    def forward(self, users, pos_items, neg_items, all_reg, all_ctrl):

        all_users, all_items = self.compute()
        all_items=torch.cat([all_items,self.embed_item_ctrl.weight])

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]
        item_embed = all_items[all_reg]
        control_embed = all_items[all_ctrl]

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)   #users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer/self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores)+1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        cf_loss = torch.sqrt(torch.sum(torch.square(torch.subtract(F.normalize(item_embed,p=2,dim=0), F.normalize(control_embed,p=2,dim=0)))))
        cf_loss = cf_loss * self.cf_pen #/ self.batch_size

        return mf_loss, reg_loss, cf_loss

class MACR(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.alpha = args.alpha
        self.beta = args.beta
        self.w = nn.Embedding(self.emb_dim, 1)
        self.w_user = nn.Embedding(self.emb_dim, 1)
        nn.init.xavier_normal_(self.w.weight)
        nn.init.xavier_normal_(self.w_user.weight)

        self.pos_item_scores = torch.empty((self.batch_size, 1))
        self.neg_item_scores = torch.empty((self.batch_size, 1))
        self.user_scores = torch.empty((self.batch_size, 1))

        self.rubi_c = args.c * torch.ones([1]).cuda(self.device)

    def forward(self, users, pos_items, neg_items):
        # Original scores
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        # print(users_emb.shape, pos_emb.shape, neg_emb.shape)
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)


        # Item module and User module
        self.pos_item_scores =torch.matmul(pos_emb, self.w.weight)
        self.neg_item_scores = torch.matmul(neg_emb, self.w.weight)
        self.user_scores = torch.matmul(users_emb, self.w_user.weight)

        # fusion
        # [batch_size,] [batch_size, 1] -> [batch_size, batch_size] * [batch_size, 1]
        # [batch_size * (bs-1)]
        pos_scores = pos_scores * torch.sigmoid(self.pos_item_scores) * torch.sigmoid(self.user_scores)
        neg_scores = neg_scores * torch.sigmoid(self.neg_item_scores) * torch.sigmoid(self.user_scores)
        #pos_scores = torch.mean(pos_scores) * torch.squeeze(torch.sigmoid(self.pos_item_scores)) * torch.squeeze(torch.sigmoid(self.user_scores))
        #neg_scores = torch.mean(neg_scores) * torch.squeeze(torch.sigmoid(self.neg_item_scores)) * torch.squeeze(torch.sigmoid(self.user_scores))

        # loss
        mf_loss_ori = torch.mean(torch.negative(torch.log(torch.sigmoid(pos_scores) + 1e-10)) + torch.negative(
            torch.log(1 - torch.sigmoid(neg_scores) + 1e-10)))

        mf_loss_item = torch.mean(
            torch.negative(torch.log(torch.sigmoid(self.pos_item_scores) + 1e-10)) + torch.negative(
                torch.log(1 - torch.sigmoid(self.neg_item_scores) + 1e-10)))

        mf_loss_user = torch.mean(torch.negative(torch.log(torch.sigmoid(self.user_scores) + 1e-10)) + torch.negative(
            torch.log(1 - torch.sigmoid(self.user_scores) + 1e-10)))

        mf_loss = mf_loss_ori + self.alpha * mf_loss_item + self.beta * mf_loss_user

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

    def predict(self, users, items=None): #未修复cosine similarity
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)

        rate_batch = torch.matmul(users, items)

        item_scores = torch.matmul(torch.transpose(items,0,1), self.w.weight)
        user_scores = torch.matmul(users, self.w_user.weight)

        rubi_rating_both = (rate_batch - self.rubi_c) * (torch.sigmoid(user_scores)) * torch.transpose(torch.sigmoid(item_scores),0,1)

        return rubi_rating_both.cpu().detach().numpy()

class SAMREG(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.rweight=args.rweight

    
    def get_correlation_loss(self,y_true, y_pred):
        x = y_true
        y = y_pred
        mx = torch.mean(x)
        my = torch.mean(y)
        xm, ym = x-mx, y-my
        r_num = torch.sum(torch.mul(xm,ym))
        r_den = torch.sqrt(torch.mul(torch.sum(torch.square(xm)), torch.sum(torch.square(ym))))
        #print(r_den)
        r = r_num / (r_den+1e-5)
        r =torch.square(torch.clamp(r,-1,1))
        return r

    def forward(self, users, pos_items, neg_items, pop_weight):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        bpr=torch.sigmoid(pos_scores - neg_scores)

        maxi = torch.log(bpr)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        mf_loss = (1-self.rweight)*(mf_loss+reg_loss)

        cor_loss=self.rweight*self.get_correlation_loss(pop_weight,bpr)

        return mf_loss, cor_loss

class INFONCE(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def forward(self, users, pos_items, neg_items):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if(self.train_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)

        

        numerator = torch.exp(pos_ratings / self.tau)

        denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1) # ssm
        # denominator = numerator + 64*torch.sum(torch.exp(neg_ratings / self.tau), dim = 1) # ssm+K
        # rand_weight = torch.rand(self.batch_size, self.neg_sample).cuda(self.device)
        # denominator = numerator + 128*torch.sum(torch.exp(neg_ratings / self.tau)*rand_weight, dim = 1) # ssm+K+rand
        
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss
    
class INFONCE_linear(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def forward(self, users, pos_items, neg_items, neg_items_pop):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if(self.train_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)

        numerator = torch.exp(pos_ratings / self.tau)

        # 对neg_items_pop取log
        # neg_items_pop_ = torch.log(neg_items_pop)
        neg_items_pop_ = neg_items_pop/neg_items_pop.max(dim=1, keepdim=True)[0]
        neg_weight = torch.softmax(neg_items_pop_.type_as(neg_ratings)/3, dim = 1)
        #@ 加入eta
        denominator = numerator + int(neg_items.shape[1]) * torch.sum(torch.exp(neg_ratings / self.tau)*neg_weight, dim = 1)
        # denominator = numerator + int(neg_items.shape[1]) * torch.sum(torch.exp(neg_ratings / self.tau)*neg_items_pop_, dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss

class AdvDRO(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.k_neg = args.k_neg
        self.w_emb_dim = args.w_embed_size
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.adv_version = args.adv_version
        self.model_version = args.model_version

        if(self.model_version == "mlp"):
            self.w_emb_dim = 4
            self.u_mlp = nn.Sequential(
                nn.Linear(self.emb_dim, self.w_emb_dim),
                nn.ReLU())
            self.i_mlp = nn.Sequential(
                nn.Linear(self.emb_dim, self.w_emb_dim),
                nn.ReLU())
        else:
            self.embed_user_p = nn.Embedding(self.n_users, self.w_emb_dim)
            self.embed_item_p = nn.Embedding(self.n_items, self.w_emb_dim)
            nn.init.xavier_normal_(self.embed_user_p.weight)
            nn.init.xavier_normal_(self.embed_item_p.weight)
        self.freeze_prob(True)

    def forward(self, users, pos_items, neg_items):

        #@ Main Branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        #@ Weight Branch
        if(self.model_version == "mlp"):
            users_p_emb = self.u_mlp(userEmb0.detach())
            neg_p_emb = self.i_mlp(negEmb0.detach())
        else:
            users_p_emb = self.embed_user_p(users)
            neg_p_emb = self.embed_item_p(neg_items)

        s_negative = torch.matmul(torch.unsqueeze(users_p_emb, 1), 
                                    neg_p_emb.permute(0, 2, 1)).squeeze(dim=1)

        users_p_emb = F.normalize(users_p_emb, dim = -1)
        neg_p_emb = F.normalize(neg_p_emb, dim = -1)

        p_negative = torch.softmax(s_negative, dim=1) # score for negative samples
        
        
        # main branch
        # use cosine similarity
        if(self.train_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)

        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)

        numerator = torch.exp(pos_ratings / self.tau)
        #@ 加入p_negative
        if(self.adv_version == 's'):
            denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1) #@ Simple SSM version
        elif(self.adv_version == 'r'):
            denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau)/p_negative, dim = 1) #@ IPS 
        elif(self.adv_version == 'pknm'):   
            denominator = numerator + self.k_neg * int(p_negative.shape[1]) * torch.sum(torch.exp(neg_ratings / self.tau)*p_negative, dim = 1) #@ multiply with N

        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        reg_neg_prob = 0.5 * torch.norm(users_p_emb) ** 2 + 0.5 * torch.norm(neg_p_emb) ** 2
        reg_neg_prob = reg_neg_prob / self.batch_size
        reg_loss_prob = self.decay * regularizer

        #@ calculate eta
        kl_d = (p_negative*torch.log(p_negative/(1/self.neg_sample))).cpu().detach().numpy()
        kl_d = np.sum(kl_d, axis=1)
        # print(kl_d.shape, type(kl_d))
        # print(max(kl_d))
        eta_u_ = {}
        for idx, u in enumerate(list(users.cpu().detach().numpy())):
            kl_d_u = kl_d[idx]
            if u not in eta_u_.keys():
                eta_u_[u] = [kl_d_u]
            else:
                eta_u_[u].append(kl_d_u)

        return ssm_loss, reg_loss, reg_loss_prob, eta_u_, p_negative

    def freeze_prob(self, flag):
        if(self.model_version == "mlp"):
            if flag:
                for param in self.u_mlp.parameters():
                    param.requires_grad = False
                for param in self.i_mlp.parameters():
                    param.requires_grad = False
                self.embed_user.requires_grad_(True)
                self.embed_item.requires_grad_(True)
            else:
                for param in self.u_mlp.parameters():
                    param.requires_grad = True
                for param in self.i_mlp.parameters():
                    param.requires_grad = True
                self.embed_user.requires_grad_(False)
                self.embed_item.requires_grad_(False)
        else:
            if flag:
                self.embed_user_p.requires_grad_(False)
                self.embed_item_p.requires_grad_(False)
                self.embed_user.requires_grad_(True)
                self.embed_item.requires_grad_(True)
            else:
                self.embed_user_p.requires_grad_(True)
                self.embed_item_p.requires_grad_(True)
                self.embed_user.requires_grad_(False)
                self.embed_item.requires_grad_(False)

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = all_items[torch.tensor(items).cuda(self.device)]

        if(self.pred_norm):
            users = F.normalize(users, dim = -1)
            items = F.normalize(items, dim = -1)

        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) # 返回一个u*i的矩阵就行 290*300 for coat 返回值需要是u*i就行

        return rate_batch.cpu().detach().numpy()

class BC_LOSS(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.n_users_pop=data.n_user_pop
        self.n_items_pop=data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)
    
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        # popularity branch
        users_pop_emb = self.embed_user_pop(users_pop)
        pos_pop_emb = self.embed_item_pop(pos_items_pop)
        neg_pop_emb = self.embed_item_pop(neg_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)

        users_pop_emb = F.normalize(users_pop_emb, dim = -1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim = -1)
        neg_pop_emb = F.normalize(neg_pop_emb, dim = -1)

        pos_ratings = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_pop_emb, 1), 
                                       neg_pop_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim = 1)
        loss2 = self.w_lambda * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if(self.pred_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        pos_ratings = torch.cos(torch.arccos(torch.clamp(pos_ratings,-1+1e-7,1-1e-7))+(1-torch.sigmoid(pos_ratings_margin)))
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim = 1)
        
        loss1 = (1-self.w_lambda) * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + \
                        0.5 * torch.norm(negEmb0) ** 2
        regularizer1 = regularizer1/self.batch_size

        regularizer2= 0.5 * torch.norm(users_pop_emb) ** 2 + 0.5 * torch.norm(pos_pop_emb) ** 2 + \
                        0.5 * torch.norm(neg_pop_emb) ** 2
        regularizer2  = regularizer2/self.batch_size
        reg_loss = self.decay * (regularizer1+regularizer2)

        reg_loss_freeze=self.decay * (regularizer2)
        reg_loss_norm=self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm

    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)

class NBC_LOSS(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.n_users_pop=data.n_user_pop
        self.n_items_pop=data.n_item_pop
        self.k_score = args.k_score
        self.k_neg = args.k_neg
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)
    
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        # popularity branch
        users_pop_emb = self.embed_user_pop(users_pop)
        pos_pop_emb = self.embed_item_pop(pos_items_pop)
        neg_pop_emb = self.embed_item_pop(neg_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)

        neg_ratings_margin = torch.matmul(torch.unsqueeze(users_pop_emb, 1), 
                                       neg_pop_emb.permute(0, 2, 1)).squeeze(dim=1)

        users_pop_emb = F.normalize(users_pop_emb, dim = -1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim = -1)
        neg_pop_emb = F.normalize(neg_pop_emb, dim = -1)

        pos_ratings = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_pop_emb, 1), 
                                       neg_pop_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim = 1)
        loss2 = self.w_lambda * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # debias_score = torch.tensor(1) - ratings
        # debias_score = neg_ratings
        # debias_score = torch.tensor(1) - torch.arccos(torch.clamp(neg_ratings,-1+1e-7,1-1e-7)) / torch.pi
        debias_score = torch.sigmoid(self.k_score * neg_ratings_margin)
        # print(debias_score[:2])

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        # pos_ratings = torch.cos(torch.arccos(torch.clamp(pos_ratings,-1+1e-7,1-1e-7))+(1-torch.sigmoid(pos_ratings_margin)))
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau1)
        # denominator = torch.sum(torch.exp(ratings / self.tau1), dim = 1)
        denominator = numerator + self.k_neg * torch.sum(debias_score*torch.exp(neg_ratings / self.tau1), dim = 1)
        # denominator = torch.sum(debias_score*torch.exp(neg_ratings / self.tau1), dim = 1)
        
        loss1 = (1-self.w_lambda) * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + \
                        0.5 * torch.norm(negEmb0) ** 2
        regularizer1 = regularizer1/self.batch_size

        regularizer2= 0.5 * torch.norm(users_pop_emb) ** 2 + 0.5 * torch.norm(pos_pop_emb) ** 2 + \
                        0.5 * torch.norm(neg_pop_emb) ** 2
        regularizer2  = regularizer2/self.batch_size
        reg_loss = self.decay * (regularizer1+regularizer2)

        reg_loss_freeze=self.decay * (regularizer2)
        reg_loss_norm=self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm

    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)


class SimpleX(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.w_neg = args.w_neg
        self.margin = args.neg_margin
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
    
    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        pos_margin_loss = 1 - pos_ratings
        neg_margin_loss = torch.mean(torch.clamp(neg_ratings - self.margin, 0, 1), dim = -1)
        
        mf_loss = torch.mean(pos_margin_loss + self.w_neg * neg_margin_loss)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

class PID(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.embed_user = Parameter(torch.FloatTensor(self.n_users, self.emb_dim))
        self.embed_item = Parameter(torch.FloatTensor(self.n_items, self.emb_dim-1))
        self.init_params(data)
        self.lr1 = nn.Linear(self.emb_dim-1, 1) 
        self.alpha_pid = args.alpha_pid
    
    def init_params(self, data):
        stdv = 1. / math.sqrt(self.embed_user.size(1))
        self.embed_user.data.uniform_(-stdv, stdv)
        self.embed_item.data.uniform_(-stdv, stdv)

        # popular part 
        item_pop = data.item_pop_idx
        # 转化为float
        item_pop = item_pop.astype(np.float32)
        # 在第一维进行扩充
        item_pop = np.expand_dims(item_pop, axis=1)
        item_pop /= item_pop.max()

        self.item_pop = Parameter(torch.from_numpy(item_pop).float()).requires_grad_(requires_grad=False)
        self.item_pop_true = Parameter(torch.from_numpy(item_pop).float()).requires_grad_(requires_grad=False)

    def forward(self, users, pos_items, neg_items):
        embed_user = self.embed_user[users]

        # 先将63 -> 1，只更新lr1
        for p in self.lr1.parameters():
            p.requires_grad = True

        optimizer1 = torch.optim.Adam(self.lr1.parameters(), lr=1e-4, weight_decay=1e-5)
        optimizer1.zero_grad()

        pid_loss = 0
        # 做预测
        item_p_emb = self.embed_item[pos_items]
        item_n_emb = self.embed_item[neg_items]
        pop_p_predict = self.lr1(item_p_emb).to(self.device)
        pop_n_predict = self.lr1(item_n_emb).to(self.device)

        # 这里的true 应该保持不变的，是真实的结果
        pop_p_true = self.item_pop_true[pos_items].to(self.device)
        pop_n_true = self.item_pop_true[neg_items].to(self.device)

        loss_func = nn.MSELoss()

        pid_loss += loss_func(pop_p_predict, pop_p_true)
        pid_loss += loss_func(pop_n_predict, pop_n_true)
        pid_loss.backward()

        optimizer1.step()

        # 训练MF
        embed_item = torch.cat([self.embed_item, self.item_pop.to(self.device)], dim=1).to(self.device)
        embed_item.retain_grad()

        item_p = embed_item[pos_items]
        item_n = embed_item[neg_items]

        p_score = torch.sum(embed_user * item_p, 1)
        n_score = torch.sum(embed_user * item_n, 1)

        # 这次是要让这个linear的变换最大化loss
        for p in self.lr1.parameters():
            p.requires_grad = False
        pid_loss = 0
        # 做预测
        item_p_emb = self.embed_item[pos_items]
        item_n_emb = self.embed_item[neg_items]
        pop_p_predict = self.lr1(item_p_emb).to(self.device)
        pop_n_predict = self.lr1(item_n_emb).to(self.device)

        # 这里的true 应该保持不变的，是真实的结果
        pop_p_true = self.item_pop_true[pos_items].to(self.device)
        pop_n_true = self.item_pop_true[neg_items].to(self.device)

        loss_func = nn.MSELoss()

        pid_loss += loss_func(pop_p_predict, pop_p_true)
        pid_loss += loss_func(pop_n_predict, pop_n_true)

        alpha = 0.8
        mf_loss = alpha * self.bpr_loss(p_score, n_score)
        pid_loss = - (1 - alpha) * pid_loss
        loss =  mf_loss + pid_loss

        regularizer = 0.5 * torch.norm(embed_user) ** 2 + self.batch_size * 0.5 * torch.norm(embed_item) ** 2 
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, pid_loss, reg_loss

    def bpr_loss(self, p_score, n_score):
        loss = torch.log(torch.sigmoid(p_score - n_score))
        loss = -loss.mean()
        return loss

    def predict(self, user, item=None):
        if item is None:
            items = list(range(self.n_items))

        item_pop = np.zeros((self.n_items, 1))
        self.item_pop = Parameter(torch.from_numpy(item_pop).float()).requires_grad_(requires_grad=False)
        items_with_0 = torch.cat([self.embed_item, self.item_pop.cuda()], dim=1).to(self.device)
        users = self.embed_user[user]
        items = items_with_0[items]
        items = torch.transpose(items, 0, 1)
        # users = all_users[torch.tensor(users).cuda(self.device)]
        # items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()

class BISER(nn.Module):
    def __init__(self, args, data):
        super(BISER, self).__init__()
        self.name = args.modeltype
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.decay = args.regs
        self.device = torch.device(args.cuda)
        self.saveID = args.saveID
        self.train_ui_matrix = torch.tensor(data.train_ui_matrix, dtype=torch.float32).to(self.device)
        self.train_iu_matrix = torch.tensor(data.train_iu_matrix, dtype=torch.float32).to(self.device)

        if args.dataset == 'Coat':
            self.hidden_dim_u = 50
            self.hidden_dim_i = 50
            self.eta_u = 0.1
            self.eta_i = 0.2
            self.reg_u = 1e-6
            self.reg_i = 1e-7
            self.batch_size_u = 4
            self.batch_size_i = 1
            self.w_u = 0.1
            self.w_i = 0.5
        elif args.dataset == 'yahoo.new':
            self.hidden_dim_u = 200
            self.hidden_dim_i = 200
            self.eta_u = 0.01
            self.eta_i = 0.05
            self.reg_u = 0
            self.reg_i = 0
            self.batch_size_u = 1
            self.batch_size_i = 1
            self.w_u = 0.9
            self.w_i = 0.1

        self.ae_u = nn.Sequential(
            nn.Linear(self.n_items, self.hidden_dim_u),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim_u, self.n_items),
            nn.Sigmoid()
        )
        self.ae_i = nn.Sequential(
            nn.Linear(self.n_users, self.hidden_dim_i),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim_i, self.n_users),
            nn.Sigmoid()
        )

    def forward(self, input_view, view2, type_input = 'i'):            
        if type_input == 'i':
            output_i = self.ae_i(input_view)
            ppscore_i = torch.clamp(output_i, 0.1, 1)
            loss_self_unbiased_i = torch.sum(input_view/ppscore_i * torch.square(1 - output_i) + (1 - input_view/ppscore_i) * torch.square(output_i))
            loss_i_u_pos_rel_i = torch.sum(input_view * torch.square(view2 - output_i))
            # 对ae_i的参数进行l2正则化
            l2_loss = []
            for module in self.ae_i:
                if type(module) is nn.Linear:
                    l2_loss.append((module.weight ** 2).sum() / 2.0)
            reg_loss_i =  self.reg_i * sum(l2_loss)

            loss_i = reg_loss_i + loss_self_unbiased_i + self.w_i*loss_i_u_pos_rel_i
            return loss_i, reg_loss_i, loss_self_unbiased_i, loss_i_u_pos_rel_i
        
        elif type_input == 'u':
            output_u = self.ae_u(input_view)
            ppscore_u = torch.clamp(output_u, 0.1, 1)
            loss_self_unbiased_u = torch.sum(input_view/ppscore_u * torch.square(1 - output_u) + (1 - input_view/ppscore_u) * torch.square(output_u))
            loss_u_i_pos_rel_u = torch.sum(input_view * torch.square(view2 - output_u))
            # 对ae_u的参数进行l2正则化
            l2_loss = []
            for module in self.ae_u:
                if type(module) is nn.Linear:
                    l2_loss.append((module.weight ** 2).sum() / 2.0)
            reg_loss_u =  self.reg_u * sum(l2_loss)
            loss_u = reg_loss_u + loss_self_unbiased_u + self.w_u*loss_u_i_pos_rel_u
            return loss_u, reg_loss_u, loss_self_unbiased_u, loss_u_i_pos_rel_u

    
    def freeze(self, type_input='all'):
        if(type_input == 'all'):
            for p in self.ae_u.parameters():
                p.requires_grad = False
            for p in self.ae_i.parameters():
                p.requires_grad = False
        elif(type_input == 'u'):
            for p in self.ae_u.parameters():
                p.requires_grad = False
            for p in self.ae_i.parameters():
                p.requires_grad = True
        elif(type_input == 'i'):
            for p in self.ae_i.parameters():
                p.requires_grad = False
            for p in self.ae_u.parameters():
                p.requires_grad = True
    
    def cal_ratings(self):
        R_u = self.ae_u(self.train_ui_matrix)
        R_i = self.ae_i(self.train_iu_matrix).T
        self.R = (R_u + R_i)/2

    # Prediction function used when evaluation
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        rate_batch = self.R[users]

        # users = self.embed_user(torch.tensor(users).cuda(self.device))
        # items = torch.transpose(self.embed_item(torch.tensor(items).cuda(self.device)), 0, 1)
        # rate_batch = torch.matmul(users, items)
        return rate_batch.cpu().detach().numpy()

class Adap_tau(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.adap_tau_beta = args.adap_tau_beta

        self.lambertw_table = torch.FloatTensor(lambertw(np.arange(-1, 1002, 1e-4))).to(self.device)
        self.register_buffer("memory_tau", torch.full((self.n_users,), 1 / 0.10))
    
    def _loss_to_tau(self, x, x_all):
        t_0 = x_all #t_0其实是reverse的
        if x is None:
            tau = t_0 * torch.ones_like(self.memory_tau, device=self.device) #如果还没有x，则直接返回t_0
        else:
            base_laberw = torch.mean(x) # x是loss
            laberw_data = torch.clamp((x - base_laberw) / self.adap_tau_beta, #t2是beta
                                    min=-np.e ** (-1), max=1000) # 这个是lambertw中间的值
            laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
            tau = (t_0 * torch.exp(-laberw_data)).detach()

        return tau

    def _update_tau_memory(self, x):
        # x: std [B]
        # y: update position [B]
        with torch.no_grad():
            x = x.detach()
            self.memory_tau = x

    def forward(self, users, pos_items, neg_items, loss_per_user, w_0, s):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if(self.train_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        if s == 0 and w_0 is not None:
            tau_user = self._loss_to_tau(loss_per_user, w_0)
            self._update_tau_memory(tau_user)

        w = torch.index_select(self.memory_tau, 0, users).detach()

        numerator = torch.exp(pos_ratings * w_0)
        # denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1)
        denominator = torch.sum(torch.exp(ratings * w.unsqueeze(1)), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        numerator_ = torch.exp(pos_ratings)
        # denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1)
        denominator_ = torch.sum(torch.exp(ratings), dim = 1)
        ssm_loss_ = torch.negative(torch.log(numerator_/denominator_))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, ssm_loss_, reg_loss, w

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = all_items[torch.tensor(items).cuda(self.device)]
        
        if(self.pred_norm):
            users = F.normalize(users, dim = -1)
            items = F.normalize(items, dim = -1)

        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()
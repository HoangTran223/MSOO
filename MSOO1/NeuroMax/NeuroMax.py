import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
from .GR import GR
from .CTR import CTR
import torch_kmeans
import logging
import sentence_transformers


class NeuroMax(nn.Module):
    def __init__(self, vocab_size, num_topics=50, num_groups=10, en_units=200, dropout=0.,
                 cluster_distribution=None, cluster_mean=None, cluster_label=None,
                 pretrained_WE=None, embed_size=200, beta_temp=0.2, is_CTR=False,
                 weight_loss_ECR=250.0, weight_loss_GR=250.0, epoch_threshold=10,
                 alpha_GR=20.0, alpha_ECR=20.0, sinkhorn_alpha = 20.0, sinkhorn_max_iter=1000, weight_loss_CTR=100.0, learn_=0,
                 weight_loss_InfoNCE=10.0, weight_loss_CL=50.0, coef_=0.5, use_MOO=1):
        super().__init__()
        self.coef_ = coef_
        self.use_MOO = use_MOO
        self.epoch_threshold = epoch_threshold
        self.learn_ = learn_
        self.lambda_1 = self.coef_
        self.lambda_2 = self.coef_
        self.lambda_3 = self.coef_
        self.lambda_4 = self.coef_

        self.weight_loss_CTR = weight_loss_CTR
        self.num_topics = num_topics
        self.num_groups = num_groups
        self.beta_temp = beta_temp
        self.is_CTR = is_CTR
        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor(
            (np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor(
            (((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)
        self.theta_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(
                torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        # Add CTR
        self.cluster_mean = nn.Parameter(torch.from_numpy(cluster_mean).float(), requires_grad=False)
        self.cluster_distribution = nn.Parameter(torch.from_numpy(cluster_distribution).float(), requires_grad=False)
        self.cluster_label = cluster_label
        if not isinstance(self.cluster_label, torch.Tensor):
            self.cluster_label = torch.tensor(self.cluster_label, dtype=torch.long, device='cuda')
        else:
            self.cluster_label = self.cluster_label.to(device='cuda', dtype=torch.long)
        
        self.map_t2c = nn.Linear(self.word_embeddings.shape[1], self.cluster_mean.shape[1], bias=False)
        self.CTR = CTR(weight_loss_CTR, sinkhorn_alpha, sinkhorn_max_iter)
        #
        
        self.topic_embeddings = torch.empty(
            (num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(
            F.normalize(self.topic_embeddings))
        
        self.encoder1 = nn.Sequential(
            nn.Linear(vocab_size, en_units),
            nn.Softplus(),
            nn.Linear(en_units, en_units),
            nn.Softplus(),
            nn.Dropout(dropout)
        )

        self.num_topics_per_group = num_topics // num_groups
        self.ECR = ECR(weight_loss_ECR, alpha_ECR, sinkhorn_max_iter)
        self.GR = GR(weight_loss_GR, alpha_GR, sinkhorn_max_iter)
        self.group_connection_regularizer = None

        # for InfoNCE
        self.prj_rep = nn.Sequential(nn.Linear(self.num_topics, 384),
                                     nn.Dropout(dropout))
        self.prj_bert = nn.Sequential()
        self.weight_loss_InfoNCE = weight_loss_InfoNCE
        self.weight_loss_CL = weight_loss_CL

    def create_group_connection_regularizer(self):
        kmean_model = torch_kmeans.KMeans(
            n_clusters=self.num_groups, max_iter=1000, seed=0, verbose=False,
            normalize='unit')
        group_id = kmean_model.fit_predict(self.topic_embeddings.reshape(
            1, self.topic_embeddings.shape[0], self.topic_embeddings.shape[1]))
        group_id = group_id.reshape(-1)
        self.group_topic = [[] for _ in range(self.num_groups)]
        for i in range(self.num_topics):
            self.group_topic[group_id[i]].append(i)

        self.group_connection_regularizer = torch.ones(
            (self.num_topics, self.num_topics), device=self.topic_embeddings.device) / 5.
        for i in range(self.num_topics):
            for j in range(self.num_topics):
                if group_id[i] == group_id[j]:
                    self.group_connection_regularizer[i][j] = 1
        self.group_connection_regularizer.fill_diagonal_(0)
        self.group_connection_regularizer = self.group_connection_regularizer.clamp(min=1e-4)
        for _ in range(50):
            self.group_connection_regularizer = self.group_connection_regularizer / \
                self.group_connection_regularizer.sum(axis=1, keepdim=True) / self.num_topics
            self.group_connection_regularizer = (self.group_connection_regularizer \
                + self.group_connection_regularizer.T) / 2.

    def get_beta(self):
        dist = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def get_representation(self, input):
        # e1 = F.softplus(self.fc11(input))
        # e1 = F.softplus(self.fc12(e1))
        # e1 = self.fc1_dropout(e1)
        e1 = self.encoder1(input)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        return theta, mu, logvar

    def encode(self, input):
        theta, mu, logvar = self.get_representation(input)
        loss_KL = self.compute_loss_KL(mu, logvar)
        return theta, loss_KL

    def get_theta(self, input):
        theta, loss_KL = self.encode(input)
        if self.training:
            return theta, loss_KL
        else:
            return theta

    def sim(self, rep, bert):
        prep = self.prj_rep(rep)
        pbert = self.prj_bert(bert)
        return torch.exp(F.cosine_similarity(prep, pbert))

    def csim(self, bow, bert):
        pbow = self.prj_rep(bow)
        pbert = self.prj_bert(bert)
        csim_matrix = (pbow@pbert.T) / (pbow.norm(keepdim=True,
                                                  dim=-1)@pbert.norm(keepdim=True, dim=-1).T)
        csim_matrix = torch.exp(csim_matrix)
        csim_matrix = csim_matrix / csim_matrix.sum(dim=1, keepdim=True)
        return -csim_matrix.log()

    def compute_loss_InfoNCE(self, rep, contextual_emb):
        if self.weight_loss_InfoNCE <= 1e-6:
            return 0.
        else:
            sim_matrix = self.csim(rep, contextual_emb)
            return sim_matrix.diag().mean() * self.weight_loss_InfoNCE

    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term +
                     logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR

    def get_loss_GR(self):
        cost = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.topic_embeddings) + 1e1 * torch.ones(self.num_topics, self.num_topics).cuda()
        loss_GR = self.GR(cost, self.group_connection_regularizer)
        return loss_GR
    
    def get_loss_CTR(self, input, indices):
        bow = input[0]
        theta, _ = self.encode(bow)
        cd_batch = self.cluster_distribution[indices]  
        cost = self.pairwise_euclidean_distance(self.cluster_mean, self.map_t2c(self.topic_embeddings))  
        loss_CTR = self.weight_loss_CTR * self.CTR(theta, cd_batch, cost)  
        return loss_CTR
    
    def create_pairs(self, batch_data, indices):
        data = batch_data  
        batch_size = data.size(0)
        device = data.device  

        idx = torch.arange(batch_size, device=device)
        idx_combinations = torch.combinations(idx, r=2)
        idx1 = idx_combinations[:, 0]
        idx2 = idx_combinations[:, 1]

        data1 = data[idx1]  
        data2 = data[idx2]  

        cluster_labels = self.cluster_label[indices].to(device)  

        labels = (cluster_labels[idx1] != cluster_labels[idx2]).float()  

        return data1, data2, labels

    def get_loss_CL(self, theta_1, theta_2, label, margin=1.0):
        euclidean_distance = nn.functional.pairwise_distance(theta_1, theta_2)
        contrastive_loss = torch.mean(
            (1-label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
        )
        return contrastive_loss

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + \
            torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    # def forward(self, indices, is_CTR, input, epoch_id=None):
    def forward(self, indices, input, epoch_id=None):
        #bow = input["data"]
        #contextual_emb = input["contextual_embed"]
        bow = input[0]
        contextual_emb = input[1]

        rep, mu, logvar = self.get_representation(bow)
        loss_KL = self.compute_loss_KL(mu, logvar)
        theta = rep
        # theta, loss_KL = self.encode(bow)

        loss_CL = 0
        if self.weight_loss_CL != 0.0:
            data1, data2, label = self.create_pairs(input[0], indices)
            data1, data2, label = data1.to('cuda'), data2.to('cuda'), label.to('cuda')
            if self.weight_loss_CL != 0.0:
                theta1, loss_KL1 = self.encode(data1)
                theta2, loss_KL2 = self.encode(data2)
                loss_CL = self.weight_loss_CL * self.get_loss_CL(theta_1=theta1, theta_2=theta2, label=label)
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(bow * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + loss_KL

        loss_ECR = self.get_loss_ECR()

        loss_InfoNCE = 0.0
        if self.weight_loss_InfoNCE != 0.0:
            loss_InfoNCE = self.compute_loss_InfoNCE(rep, contextual_emb)
            
        #CTR

        if self.weight_loss_CTR != 0:
            loss_CTR = self.get_loss_CTR(input, indices)
        else:
            loss_CTR = 0.0
        if epoch_id == 10 and self.group_connection_regularizer is None:
            self.create_group_connection_regularizer()
        loss_GR = torch.tensor(0.0, requires_grad=False).to('cuda')
        if self.group_connection_regularizer is not None and epoch_id > 10:
            loss_GR = self.get_loss_GR()


        loss = loss_TM + loss_ECR + loss_GR + loss_InfoNCE
        # loss = loss_TM + loss_ECR + loss_GR + loss_CTR + loss_InfoNCE + loss_CL
        # loss = loss_TM + loss_ECR + loss_GR + loss_InfoNCE + loss_CL
        # loss = loss_TM + loss_ECR + loss_GR + loss_InfoNCE + loss_CTR
        # loss = loss_TM + loss_ECR + loss_GR + loss_InfoNCE
        # rst_dict = {
        #     'loss': loss,
        #     'loss_CTR': loss_CTR,
        #     'loss_TM': loss_TM,
        #     'loss_ECR': loss_ECR,
        #     'loss_GR': loss_GR,
        #     'loss_InfoNCE': loss_InfoNCE,
        # }
        if self.use_MOO == 1:
            if self.weight_loss_CTR == 0:
                if self.learn_ == 0:
                    rst_dict = {
                        'loss_': loss,
                        #'loss_CTR': loss_CTR,
                        'loss_x1': loss_TM + self.coef_ * loss,
                        'loss_x2': loss_ECR + self.coef_ * loss,
                        'loss_x3': loss_GR + self.coef_ * loss,
                        'loss_x4': loss_InfoNCE + self.coef_ * loss,
                        'lossTM': loss_TM,
                        'lossECR': loss_ECR,
                        'lossGR': loss_GR,
                        'lossInfoNCE': loss_InfoNCE,
                    }
                else:
                    rst_dict = {
                        'loss_': loss,
                        #'loss_CTR': loss_CTR,
                        'loss_x1': loss_TM + self.lambda_1 * loss,
                        'loss_x2': loss_ECR + self.lambda_2 * loss,
                        'loss_x3': loss_GR + self.lambda_3 * loss,
                        'loss_x4': loss_InfoNCE + self.lambda_4 * loss,
                        'losssTM': loss_TM,
                        'losssECR': loss_ECR,
                        'losssGR': loss_GR,
                        'losssInfoNCE': loss_InfoNCE,
                    }
            else:
                rst_dict = {
                    'loss_': loss,
                    #'loss_CTR': loss_CTR,
                    'loss_x1': loss_TM + loss_ECR + loss_GR + loss_InfoNCE + self.coef_ * loss_CTR,
                    'loss_x2': loss_TM + loss_ECR + loss_GR + self.coef_ * loss_InfoNCE + loss_CTR,
                    'loss_x3': loss_TM + loss_ECR + self.coef_ * loss_GR + loss_InfoNCE + loss_CTR,
                    'loss_x4': loss_TM + self.coef_ * loss_ECR + loss_GR + loss_InfoNCE + loss_CTR,
                    'loss_x5': self.coef_ * loss_TM + loss_ECR + loss_GR + loss_InfoNCE + loss_CTR,
                    'lossTM': loss_TM,
                    'lossECR': loss_ECR,
                    'lossGR': loss_GR,
                    'lossInfoNCE': loss_InfoNCE,
                }
        else:
            rst_dict = {
                'loss_': loss,
                #'loss_CTR': loss_CTR,
                'lossTM': loss_TM,
                'lossECR': loss_ECR,
                'lossGR': loss_GR,
                'lossInfoNCE': loss_InfoNCE,
            }

        return rst_dict
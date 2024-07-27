import copy
import torch
import gc
import math
import torch.nn as nn
from fast_pytorch_kmeans import KMeans

def weight_fed(global_protos, local_protos, opt):
    protos_bank = []
    for k, v in local_protos.items():
        if bool(torch.isnan(v[0]).any()):
            continue
        else:
            protos_bank.append(v[0])

    protos_bank = torch.cat(protos_bank, dim=0).contiguous()
    protos_bank = protos_bank.cuda()
    protos_bank = torch.cat((global_protos, protos_bank), dim=0)

    kmeans = KMeans(n_clusters=global_protos.shape[0], mode='cosine')
    cluster_r = kmeans.fit_predict(protos_bank)
    cluster_global = cluster_r[0:global_protos.shape[0]]
    cluster_local = cluster_r[global_protos.shape[0]:len(cluster_r)]

    ones = torch.sparse.torch.eye(global_protos.shape[0]).cuda()
    mm_global = ones.index_select(0, cluster_global)

    modal_w = torch.zeros(1, len(local_protos))
    softMax = nn.Softmax(dim=1)
    for k, v in local_protos.items():
        cluster_label = cluster_local[k * global_protos.shape[0]: (k+1) * global_protos.shape[0]].cuda()
        mm_local = ones.index_select(0, cluster_label)

        theta = 1.0 / 2 * torch.matmul(global_protos, v[0].cuda().t())
        S = torch.matmul(mm_global, mm_local.t())
        if opt.use_gpu:
            S = S.cuda()

        modal_w[0][k] = torch.log(-torch.sum(S * theta - torch.log(1.0 + torch.exp(theta))))

    modal_w = softMax(modal_w)
    return modal_w

def weighted_agg_fed(w, modal_w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * modal_w[0][0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * modal_w[0][i]

    return w_avg




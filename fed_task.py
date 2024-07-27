import torch
from tqdm import tqdm
from utils import *
from torch.autograd import Variable
import numpy as np
import gc
import copy
from main import *
from fed_utils import *
from update import LocalImgUpdate, LocalTxtUpdate
import os
from fast_pytorch_kmeans import KMeans
import torch.nn.functional as F

def generate_prototypes(models_img, models_txt, local_train_x, local_train_y, local_train_L, opt, global_protos):
    """
    Generate prototypes

    Returns
        code(torch.Tensor): prototypes.
    """

    with torch.no_grad():
        models_img.eval()
        models_txt.eval()

        proj_bank_fuse = []
        num_train = local_train_L.shape[0]
        index = np.random.permutation(num_train)
        for i in tqdm(range(num_train // opt.batch_size + 1)):
            remaining = num_train - i * opt.batch_size
            if remaining <= 0:
                break
            ind = index[i * opt.batch_size: (i + 1) * opt.batch_size]

            sample_X = Variable(local_train_x[ind].type(torch.float))
            sample_Y = local_train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            sample_Y = Variable(sample_Y)

            if opt.use_gpu:
                sample_X = sample_X.cuda()
                sample_Y = sample_Y.cuda()
            enhanced_f_x, _, _ = models_img(sample_X, global_protos)
            enhanced_f_x = enhanced_f_x.to('cpu')
            enhanced_f_y, _, _ = models_txt(sample_Y, global_protos)
            enhanced_f_y = enhanced_f_y.to('cpu')
            H = (enhanced_f_x + enhanced_f_y) / 2
            H = F.normalize(H)
            proj_bank_fuse.append(H)
            proj_bank_img = torch.cat(proj_bank_fuse, dim=0).contiguous()

        cluster_result = {'inst2cluster': [], 'centroids': []}
        kmeans = KMeans(n_clusters=local_train_L.shape[1], mode='cosine')
        cluster_r = kmeans.fit_predict(proj_bank_img)
        cc = kmeans.centroids
        cluster_result['centroids'].append(cc)
        prototypes = cluster_result['centroids']

    torch.cuda.empty_cache()
    return prototypes


def global_proto_cluster(local_protos_dict, shape):

    protos_bank = []
    for k,v in local_protos_dict.items():
        if bool(torch.isnan(v[0]).any()):
            continue
        else:
            protos_bank.append(v[0])

    protos_bank = torch.cat(protos_bank, dim=0).contiguous()
    protos_bank = F.normalize(protos_bank, dim=1)

    kmeans = KMeans(n_clusters=shape, mode='cosine')
    cluster_r = kmeans.fit_predict(protos_bank)
    cc = kmeans.centroids
    return cc

def Fed_taskheter(models_img, models_txt, opt, X, Y, L, user_groups, server_img_model, server_txt_model, local_F_buffer_dict, local_G_buffer_dict, local_B_dict):
    train_L = torch.from_numpy(L['train']).cuda()
    train_x = torch.from_numpy(X['train']).cuda()
    train_y = torch.from_numpy(Y['train']).cuda()

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    max_mapi2t = max_mapt2i = 0.

    global_protos = torch.zeros([train_L.shape[1], opt.bit])

    if opt.use_gpu:
        global_protos = global_protos.cuda()

    for r in range(opt.rounds):
        local_protos = {}
        w_img_glob, w_txt_glob = [], []
        for j in range(opt.num_users):
            # update local weights every round
            models_img[j] = copy.deepcopy(server_img_model)
            models_txt[j] = copy.deepcopy(server_txt_model)

            # pick up the certain client's training image, text and label dataset
            idxs = user_groups[j]
            local_train_x = train_x[idxs]
            local_train_y = train_y[idxs]
            local_train_L = train_L[idxs]

            Sim = calc_neighbor(opt, local_train_L, local_train_L)

            for epoch in range(opt.max_epoch):
                # train image net
                local_img_model = LocalImgUpdate(opt=opt, F_buffer=local_F_buffer_dict[j], G_buffer=local_G_buffer_dict[j],
                                                 B=local_B_dict[j], train_img=local_train_x, train_L=local_train_L, train_txt=local_train_y)
                local_F_buffer = local_img_model.update_weights_het(global_protos, models_img[j], server_img_model, server_txt_model)
                local_F_buffer_dict[j] = local_F_buffer

                # train text net
                local_txt_model = LocalTxtUpdate(opt=opt, F_buffer=local_F_buffer_dict[j], G_buffer=local_G_buffer_dict[j],
                                                 B=local_B_dict[j], train_txt=local_train_y, train_L=local_train_L, train_img=local_train_x)
                local_G_buffer = local_txt_model.update_weights_het(global_protos, models_txt[j], server_img_model, server_txt_model)
                local_G_buffer_dict[j] = local_G_buffer

                # update B
                local_B = torch.sign(local_F_buffer + local_G_buffer)
                local_B_dict[j] = local_B

                # calculate total loss
                loss = calc_loss(local_B, local_F_buffer, local_G_buffer, Variable(Sim), opt)
                print('...round: %3d, client: %3d, epoch: %3d, loss: %3.3f' % (r + 1, j + 1, epoch + 1, loss.data))

            w_img = models_img[j].state_dict()
            w_img_glob.append(w_img)
            w_txt = models_txt[j].state_dict()
            w_txt_glob.append(w_txt)
            agg_protos = generate_prototypes(models_img[j], models_txt[j], local_train_x, local_train_y, local_train_L, opt, global_protos)

            local_protos[j] = copy.deepcopy(agg_protos)

        global_protos = global_proto_cluster(local_protos, train_L.shape[1])

        if opt.use_gpu:
            global_protos = global_protos.cuda()

        modal_w = weight_fed(global_protos, local_protos, opt)

        # calculate updated global weights
        w_img_new = weighted_agg_fed(w_img_glob, modal_w)
        w_txt_new = weighted_agg_fed(w_txt_glob, modal_w)
        server_img_model.load_state_dict(w_img_new)
        server_txt_model.load_state_dict(w_txt_new)

        # send updated global weights to every client
        for i in range(opt.num_users):
            models_img[i] = copy.deepcopy(server_img_model)
            models_txt[i] = copy.deepcopy(server_txt_model)

        if opt.valid:
            mapi2t, mapt2i = valid(server_img_model, server_txt_model, query_x, retrieval_x, query_y, retrieval_y,
                                       query_L, retrieval_L, opt, global_protos)
            print('...round: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (r + 1, mapi2t, mapt2i))

            if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i

    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    else:
        mapi2t, mapt2i = valid(server_img_model, server_txt_model, query_x, retrieval_x, query_y, retrieval_y,
                               query_L, retrieval_L, opt, global_protos)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))



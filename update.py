import numpy as np
import torch
from tqdm import tqdm
from utils import *
from torch.autograd import Variable
import torch.nn as nn
import copy

class LocalImgUpdate(object):
    def __init__(self, opt, F_buffer, G_buffer, B, train_img, train_L, train_txt):
        self.opt = opt
        self.F_buffer = F_buffer
        self.G_buffer = G_buffer
        self.B = B
        self.train_image = train_img
        self.train_L = train_L
        self.train_text = train_txt
        self.Sim = calc_neighbor(opt, train_L, train_L)
        self.ones = torch.ones(opt.batch_size, 1).cuda()

    def update_weights_het(self, global_protos, model, server_img_model, server_txt_model):
        # set mode to train model
        num_train = self.train_image.shape[0]
        batch_size = self.opt.batch_size
        ones_ = torch.ones(num_train - batch_size, 1).cuda()
        optimizer_img = torch.optim.SGD(model.parameters(), lr=self.opt.lr)
        server_img_model.eval()
        server_txt_model.eval()
        model.train()

        # train image net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(self.train_L[ind, :])
            image = Variable(self.train_image[ind].type(torch.float))
            text = self.train_text[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text)
            if self.opt.use_gpu:
                text = text.cuda()
                image = image.cuda()
                sample_L = sample_L.cuda()
                ones = self.ones.cuda()
                ones_ = ones_.cuda()

            S = calc_neighbor(self.opt, sample_L, self.train_L)

            enhanced_f, l_predict, cur_f = model(image, global_protos)
            enhanced_f_gy, l_predict_gy, _ = server_txt_model(text, global_protos)
            enhanced_f_gx, l_predict_gx, _ = server_img_model(image, global_protos)
            self.F_buffer[ind, :] = enhanced_f.data
            F = Variable(self.F_buffer)
            G = Variable(self.G_buffer)

            # classification loss
            predict_loss = nn.functional.kl_div(l_predict.log(), sample_L, reduction='mean')

            # feature constraint loss
            cos = nn.CosineEmbeddingLoss(reduction='mean')
            loss_flag = torch.ones([batch_size]).cuda()
            feature_loss = cos(cur_f.float(), enhanced_f.float(), loss_flag)

            # cosine similarity loss
            cos_x = cos(enhanced_f.float(), enhanced_f_gx.float(), loss_flag)
            cos_y = cos(enhanced_f.float(), enhanced_f_gy.float(), loss_flag)
            cos_loss = 0.5 * (cos_x + cos_y)

            # hash loss
            theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
            logloss_x = -torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
            quantization_x = torch.sum(torch.pow(self.B[ind, :] - cur_f, 2))
            balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))
            loss_x = logloss_x + self.opt.gamma * quantization_x + self.opt.alphard * balance_x
            loss_x /= (num_train * batch_size)
            loss_x = (1.0 / self.opt.num_users)*loss_x

            # overall loss
            loss_x = loss_x + self.opt.alpha * feature_loss + self.opt.eta * cos_loss + self.opt.mu * predict_loss

            if (torch.isnan(loss_x).any()):
                continue

            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()

        return self.F_buffer


class LocalTxtUpdate(object):
    def __init__(self, opt, F_buffer, G_buffer, B, train_txt, train_L, train_img):
        self.opt = opt
        self.F_buffer = F_buffer
        self.G_buffer = G_buffer
        self.B = B
        self.train_text = train_txt
        self.train_image = train_img
        self.train_L = train_L
        self.Sim = calc_neighbor(opt, train_L, train_L)
        self.ones = torch.ones(opt.batch_size, 1).cuda()

    def update_weights_het(self, global_protos, model, server_img_model, server_txt_model):
        # set mode to train model
        num_train = self.train_text.shape[0]
        batch_size = self.opt.batch_size
        ones_ = torch.ones(num_train - batch_size, 1).cuda()
        optimizer_txt = torch.optim.SGD(model.parameters(), lr=self.opt.lr)
        server_img_model.eval()
        server_txt_model.eval()
        model.train()

        # train text net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(self.train_L[ind, :])
            image = Variable(self.train_image[ind].type(torch.float))
            text = self.train_text[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text)
            if self.opt.use_gpu:
                text = text.cuda()
                image = image.cuda()
                sample_L = sample_L.cuda()
                ones = self.ones.cuda()
                ones_ = ones_.cuda()

            # similar matrix
            S = calc_neighbor(self.opt, sample_L, self.train_L)

            enhanced_f, l_predict, cur_g= model(text, global_protos)
            enhanced_f_gy, l_predict_y, _ = server_txt_model(text, global_protos)
            enhanced_f_gx, l_predict_x, _ = server_img_model(image, global_protos)
            self.G_buffer[ind, :] = enhanced_f.data
            F = Variable(self.F_buffer)
            G = Variable(self.G_buffer)

            # classification loss
            predict_loss = nn.functional.kl_div(l_predict.log(), sample_L, reduction='mean')

            # feature constraint loss
            cos = nn.CosineEmbeddingLoss(reduction='mean')
            loss_flag = torch.ones([batch_size]).cuda()
            feature_loss = cos(cur_g.float(), enhanced_f.float(), loss_flag)

            # cosine similarity loss
            cos_x = cos(enhanced_f.float(), enhanced_f_gx.float(), loss_flag)
            cos_y = cos(enhanced_f.float(), enhanced_f_gy.float(), loss_flag)
            cos_loss = 0.5 * (cos_x + cos_y)

            # hash loss
            theta_y = 1.0 / 2 * torch.matmul(cur_g, F.t())
            logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
            quantization_y = torch.sum(torch.pow(self.B[ind, :] - cur_g, 2))
            balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G[unupdated_ind].t().mm(ones_), 2))
            loss_y = logloss_y + self.opt.gamma * quantization_y + self.opt.alphard * balance_y
            loss_y /= (num_train * batch_size)
            loss_y = (1.0 / self.opt.num_users) * loss_y

            # overall loss
            loss_y = loss_y + self.opt.alpha * feature_loss + self.opt.eta * cos_loss + self.opt.mu * predict_loss

            if (torch.isnan(loss_y).any()):
                continue

            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()

        return self.G_buffer


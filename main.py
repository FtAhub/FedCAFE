from data_handler import *
from models import ImgModule, TxtModule
from utils import *
import copy
import random
from options import args_parser
import os
import gc
from fed_task import *

def train(opt):
    # loading and splitting data
    images, tags, labels = load_data(opt.data_path)
    X, Y, L = split_data(images, tags, labels, opt)

    # sample IID client data from dataset
    user_groups = iid(opt, L['train'])
    print('...loading and splitting data finish')

    # build models
    y_dim = Y['train'].shape[1]
    num_prototypes = L['train'].shape[1]
    pretrain_model = load_pretrain_model(opt.pretrain_model_path)
    server_img_model = ImgModule(opt.bit, pretrain_model, num_prototypes)
    server_txt_model = TxtModule(y_dim, opt.bit, num_prototypes)

    if opt.use_gpu:
        server_img_model = server_img_model.cuda()
        server_txt_model = server_txt_model.cuda()

    print('...Structure initialization is completed...')

    F_buffer = {}
    G_buffer = {}
    B = {}

    models_img, models_txt = [], []
    for client in range(opt.num_users):
        model_img = copy.deepcopy(server_img_model)
        model_txt = copy.deepcopy(server_txt_model)
        models_img.append(model_img)
        models_txt.append(model_txt)
        num_train = len(user_groups[client])
        F_buffer[client] = torch.randn(num_train, opt.bit)
        G_buffer[client] = torch.randn(num_train, opt.bit)
        B[client] = torch.sign(F_buffer[client] + G_buffer[client])
        if opt.use_gpu:
            F_buffer[client] = F_buffer[client].cuda()
            G_buffer[client] = G_buffer[client].cuda()
            B[client] = B[client].cuda()

    # training start
    Fed_taskheter(models_img, models_txt, opt, X, Y, L, user_groups, server_img_model, server_txt_model, F_buffer, G_buffer, B)


def valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L, opt, global_protos):
    qBX = generate_image_code(img_model, query_x, opt, global_protos)
    qBY = generate_text_code(txt_model, query_y, opt, global_protos)
    rBX = generate_image_code(img_model, retrieval_x, opt, global_protos)
    rBY = generate_text_code(txt_model, retrieval_y, opt, global_protos)
    if opt.use_gpu:
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()
    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    return mapi2t, mapt2i


def test(opt):
    images, tags, labels = load_data(opt.data_path)
    X, Y, L = split_data(images, tags, labels, opt)
    # sample IID client data from dataset
    print('...loading and splitting data finish')

    # build models
    y_dim = Y['train'].shape[1]
    num_prototypes = L['train'].shape[1]
    img_model = ImgModule(bit=opt.bit, pretrain_model=None, num_prototypes=num_prototypes)
    txt_model = TxtModule(y_dim=y_dim, bit=opt.bit, num_prototypes=num_prototypes)
    global_protos = h5py.File(opt.load_grobal_memory)['memory'][:]

    if opt.load_img_path:
        img_model.load(opt.load_img_path)

    if opt.load_txt_path:
        txt_model.load(opt.load_txt_path)

    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    qBX = generate_image_code(img_model, query_x, opt, global_protos)
    qBY = generate_text_code(txt_model, query_y, opt, global_protos)
    rBX = generate_image_code(img_model, retrieval_x, opt, global_protos)
    rBY = generate_text_code(txt_model, retrieval_y, opt, global_protos)

    if opt.use_gpu:
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    print('...test MAP: MAP(i->t): %3.3f, MAP(t->i): %3.3f' % (mapi2t, mapt2i))

def generate_image_code(img_model, X, opt, global_protos):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, opt.bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float)
        if opt.use_gpu:
            image = image.cuda()
        cur_f, _, _ = img_model(image, global_protos)
        B[ind, :] = cur_f.data
    B = torch.sign(B)
    return B


def generate_text_code(txt_model, Y, opt, global_protos):
    batch_size = opt.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, opt.bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
        if opt.use_gpu:
            text = text.cuda()
        cur_g, _, _ = txt_model(text, global_protos)
        B[ind, :] = cur_g.data
    B = torch.sign(B)
    return B



def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --lr=0.01
            python {0} help
    avaiable args:'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print('\t\t{0}: {1}'.format(k, v))


if __name__ == '__main__':
    # default parameter settings
    opt = args_parser()

    # set random seeds
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if opt.device == 'cuda':
        torch.cuda.set_device(opt.gpu)
        torch.cuda.manual_seed(opt.seed)
        torch.manual_seed(opt.seed)
    else:
        torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    #training
    train(opt)
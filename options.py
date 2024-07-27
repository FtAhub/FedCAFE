import warnings
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # load model path
    load_img_path = None
    load_txt_path = None
    load_grobal_memory = None

    # data parameters
    parser.add_argument('--data_path', type=str, default='./data/FLICKR-25K.mat',
                        help="directory of dataset")
    parser.add_argument('--pretrain_model_path', type=str, default='./data/imagenet-vgg-f.mat',
                        help="directory of pretrain_model")
    parser.add_argument('--training_size', type=int, default=10000,
                        help="training_size")
    parser.add_argument('--query_size', type=int, default=2000,
                        help="query_size")
    parser.add_argument('--database_size', type=int, default=18015,
                        help="database_size")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="batch_size"),

    # federated arguments
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")

    # hyper-parameters
    parser.add_argument('--max_epoch', type=int, default=20,
                        help="the number of local episodes: E")
    parser.add_argument('--lr', type=float, default=10 ** (-1.5),
                        help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=1,
                        help='gamma')
    parser.add_argument('--alphard', type=float, default=1,
                        help='alphard')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='alpha')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='eta')
    parser.add_argument('--mu', type=float, default=1,
                        help='mu')
    parser.add_argument('--bit', type=int, default=32,
                        help='final binary code length')

    parser.add_argument('--use_gpu', default=1,
                        help="To use cuda, default set to use CPU.")
    parser.add_argument('--gpu', default=0,
                        help="Set to a specific GPU ID.")
    parser.add_argument('--valid', default=1,
                        help="valid")
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

    opt = parser.parse_args()
    return opt




from __future__ import print_function, absolute_import
import os
import argparse
import random
import numpy as np
# torch-related packages
import torch
import matplotlib.pyplot as plt
from utils.visualization import visualize_TSNE
from utils.myfuncs import set_determinism, set_logger
from datetime import datetime

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# data
from data_loader import Visda_Dataset, Office_Dataset, Home_Dataset, Visda18_Dataset
from model_trainer import ModelTrainer
from utils.logger import Logger

def main(args):

    # total_step = int(100//args.EF)
    total_step = int(100//args.EF)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # prepare checkpoints and log folders
    set_determinism()
    args.experiment = set_exp_name(args)
    args.logs_dir = os.path.join('logs', args.dataset, args.experiment)
    args.checkpoints_dir = os.path.join(args.logs_dir, 'checkpoints')
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    # initialize dataset
    if args.dataset == 'visda':
        args.data_dir = os.path.join(args.data_dir, 'visda')
        data = Visda_Dataset(root=args.data_dir, partition='train', label_flag=None)

    elif args.dataset == 'office':
        args.data_dir = os.path.join(args.data_dir, 'Office31', 'imgs')
        data = Office_Dataset(root=args.data_dir, txt_root=args.txt_root, partition='train', label_flag=None, source=args.source_name,
                              target=args.target_name)

    elif args.dataset == 'home':
        args.data_dir = os.path.join(args.data_dir, 'OfficeHome', 'imgs')
        data = Home_Dataset(root=args.data_dir, txt_root=args.txt_root, partition='train', label_flag=None, source=args.source_name,
                              target=args.target_name)
    elif args.dataset == 'visda18':
        args.data_dir = os.path.join(args.data_dir, 'visda18')
        data = Visda18_Dataset(root=args.data_dir, partition='train', label_flag=None)
    else:
        print('Unknown dataset!')

    args.class_name = data.class_name
    args.num_class = data.num_class
    args.alpha = data.alpha
    # setting experiment name
    label_flag = None
    selected_idx = None

    os.makedirs(args.logs_dir, exist_ok=True)
    set_logger(args.logs_dir)
    logger = Logger(args)

    if not args.visualization:

        for step in range(total_step):

            print("This is {}-th step with EF={}%".format(step, args.EF))

            trainer = ModelTrainer(args=args, data=data, step=step, label_flag=label_flag, v=selected_idx, logger=logger)

            # train the model
            args.log_epoch = 4 + step//2
            trainer.train(step, epochs= 4 + (step) * 2, step_size=args.log_epoch)

            # pseudo_label
            pred_y, pred_score, pred_acc = trainer.estimate_label()

            # select data from target to source
            selected_idx = trainer.select_top_data(pred_score)

            # add new data
            label_flag, data = trainer.generate_new_train_data(selected_idx, pred_y, pred_acc)
    else:
        # load trained weights
        trainer = ModelTrainer(args=args, data=data)
        trainer.load_model_weight(args.checkpoint_path)
        vgg_feat, node_feat, target_labels, split = trainer.extract_feature()
        visualize_TSNE(node_feat, target_labels, args.num_class, args, split)

        plt.savefig('./node_tsne.png', dpi=300)



def set_exp_name(args):
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    exec_num = os.environ['exec_num'] if 'exec_num' in os.environ.keys() else 0
    now  = datetime.now().strftime('%y%m%d_%H:%M:%S')
    exp_name  = now
    exp_name += f'--c{gpu[0]}n{exec_num}'
    exp_name += f'--{args.dset}'
    # exp_name += f'--{args.source_name}{args.target_name}'
    exp_name += f'--{args.task}'
    exp_name += '--A{}'.format(args.arch)
    exp_name += '_L{}'.format(args.num_layers)
    exp_name += '_E{}_B{}'.format(args.EF, args.batch_size)
    return exp_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progressive Graph Learning for Open-set Domain Adaptation')
    # set up dataset & backbone embedding
    # parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('-a', '--arch', type=str, default='res')
    parser.add_argument('--root_path', type=str, default='./utils/', metavar='B',
                        help='root dir')

    # set up path
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='/nas/data/syamagami/GDA/data')
    # parser.add_argument('--logs_dir', type=str, metavar='PATH',
    #                     default=os.path.join(working_dir, 'logs'))
    # parser.add_argument('--checkpoints_dir', type=str, metavar='PATH',
    #                     default=os.path.join(working_dir, 'checkpoints'))

    # verbose setting
    parser.add_argument('--log_step', type=int, default=30)
    parser.add_argument('--log_epoch', type=int, default=3)

    parser.add_argument('--dset', type=str, default='amazon_dslr')
    parser.add_argument('--task', type=str, default='true_domains')
    parser.add_argument('--dataset', type=str, default='home', choices=['home', 'office', 'visda', 'visda18'])

    # if dataset == 'office':
    #     parser.add_argument('--source_name', type=str, default='D')
    #     parser.add_argument('--target_name', type=str, default='W')

    # elif dataset == 'home':
    #     parser.add_argument('--source_name', type=str, default='R')
    #     parser.add_argument('--target_name', type=str, default='A')
    # else:
    #     print("Set a log step first !")
    parser.add_argument('--eval_log_step', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=1500)

    # hyper-parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('-b', '--batch_size', type=int, default=1)  # 元々は4
    # parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.6)  ### TODO: これは beta であろうか. 論文のTable 1の設定であろうか. もともとは0.1

    parser.add_argument('--dropout', type=float, default=0.2)
    # parser.add_argument('--EF', type=int, default=10)
    parser.add_argument('--EF', type=int, default=0.1)  ### TODO: Enlarging Factor α. もともとは10. 論文のTable 1の設定. 
    parser.add_argument('--loss', type=str, default='focal')


    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    # GNN parameters
    parser.add_argument('--in_features', type=int, default=2048)
    # if dataset == 'home':
    #     parser.add_argument('--node_features', type=int, default=512)
    #     parser.add_argument('--edge_features', type=int, default=512)
    # else:
    #     parser.add_argument('--node_features', type=int, default=1024)
    #     parser.add_argument('--edge_features', type=int, default=1024)

    parser.add_argument('--num_layers', type=int, default=1)

    #tsne
    parser.add_argument('--visualization', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='/home/Desktop/Open_DA_git/checkpoints/D-visda18_A-res_L-1_E-20_B-4_step_1.pth.tar')

    #Discrminator
    parser.add_argument('--discriminator', type=bool, default=True)
    parser.add_argument('--adv_coeff', type=float, default=0.4)

    #GNN hyper-parameters
    parser.add_argument('--node_loss', type=float, default=0.3)

    args = parser.parse_args()
    
    args.txt_root = f'data/{args.dataset}/{args.task}'
    args.source_name = args.dset.split('_')[0][0]
    args.target_name = args.dset.split('_')[1][0]

    if args.dataset == 'home':
        args.node_features = 512
        args.edge_features = 512
    else:
        args.node_features = 1024
        args.edge_features = 1024

    
    main(args)


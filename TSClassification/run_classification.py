import argparse
import os
import torch
from exp.exp_classification import Exp_Classification
from pprint import pprint
from utils.print_args import print_args
import random
import numpy as np

parser = argparse.ArgumentParser(description='ConvTimeNet')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model', type=str, required=False, default='ConvTimeNet')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--root_path', type=str, required=True)

parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')

parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--e_layers', type=int, default=None, help='num of encoder layers')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)

parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')

# ConvTimeNet
parser.add_argument('--patch_size', type=int, default=16, help='patch size of deformable patch module')
parser.add_argument('--patch_stride', type=int, default=None, help='patch stride of deformable patch module')
parser.add_argument('--dw_ks', type=str, default='7,7,13,13,19,19', help="kernel size for each deep-wise convolution")

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--lr_decay', type=int, default=5, help="0.9*lr per N epochs")
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')




args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

args.patch_stride = int(0.5 * args.patch_size) if args.patch_stride == None else args.patch_stride

fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# convert string to list
args.dw_ks = [int(ks) for ks in args.dw_ks.split(',')]
args.e_layers = len(args.dw_ks)

torch.cuda.set_device(0)

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
# print_args(args)
pprint(args)

Exp = Exp_Classification

accs, f1s, precs, recalls = [], [], [], []
if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        exp = Exp(args)  # set experiments
        setting = '{}_{}_dm{}_ps{}_sd{}_dw{}_dr{}_lr{}_lrd{}_el{}_df{}_dt{}_{}_{}'.format(
            args.model,
            args.dataset,
            args.d_model,
            args.patch_size,
            args.patch_stride,
            args.dw_ks,
            args.dropout,
            args.learning_rate,
            args.lr_decay,
            args.e_layers,
            args.d_ff,
            args.patch_size,
            args.distil,
            args.des, ii)

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        accuracy, f1, precision, recall = exp.test(setting)
        accs.append(accuracy); f1s.append(f1)
        precs.append(precision), recalls.append(recall)
        torch.cuda.empty_cache()
        
    print('\n============= AVERAGE =============')
        
    print('average acc:{0:.4f}±{1:.4f}, f1:{2:.4f}±{3:.4f}'.format(np.mean(
        accs), np.std(accs), np.mean(f1s), np.std(f1s)))
    
    print('average precision:{0:.4f}±{1:.4f}, recall:{2:.4f}±{3:.4f}'.format(np.mean(
        precs), np.std(precs), np.mean(recalls), np.std(recalls)))
    
    # # save the results
    # file_dir = f"results"
    # file_path = os.path.join(file_dir, f"{args.dataset}.txt")
    
    # if not os.path.exists(file_path):
    #     os.makedirs(file_dir, exist_ok=True)
    #     with open(file_path, 'w') as f:
    #         pass
        
    # with open(file_path, 'a') as f:
    #     f.write(setting + '\n')
    #     f.write('accs:{}, f1:{}'.format(accs, f1s))
         
    #     f.write('\n\n')
    
else:
    ii = 0
    setting = '{}_{}_dm{}_ps{}_sd{}_dw{}_dr{}_lr{}_lrd{}_el{}_df{}_dt{}_{}_{}'.format(
        args.model,
        args.dataset,
        args.d_model,
        args.patch_size,
        args.patch_stride,
        args.dw_ks,
        args.dropout,
        args.learning_rate,
        args.lr_decay,
        args.e_layers,
        args.d_ff,
        args.patch_size,
        args.distil,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()

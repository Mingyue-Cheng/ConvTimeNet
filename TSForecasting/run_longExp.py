import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import sys
os.chdir(sys.path[0])
parser = argparse.ArgumentParser(description='ConvTimeNet for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='DePatchConv',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# ConvTimeNet
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')

parser.add_argument('--dw_ks', type=str, default='11,15,21,29,39,51', help="kernel size of the deep-wise. default:9")
parser.add_argument('--re_param', type=int, default=1, help='Reparam the DeepWise Conv when train')
parser.add_argument('--enable_res_param', type=int, default=1, help='Learnable residual')
parser.add_argument('--re_param_kernel', type=int, default=3)

# Patch
parser.add_argument('--patch_ks', type=int, default=32, help="kernel size of the patch window. default:32")
parser.add_argument('--patch_sd', type=float, default=0.5, \
					help="stride of the patch window. default: 0.5. if < 1, then sd = patch_sd * patch_ks")


# Other Parameter
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') 
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')

parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--CTX', type=str, default='0', required=False, help='visuable device ids')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.CTX

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

args.dw_ks = [int(ks) for ks in args.dw_ks.split(',')]
args.patch_sd = max(1, int(args.patch_ks * args.patch_sd)) if args.patch_sd <= 1 else int(args.patch_sd)
assert args.e_layers == len(args.dw_ks), "e_layers should match the dw kernel list!"

print('Args in experiment:')
print(args)

Exp = Exp_Main
mses = []
maes = []
if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_dm{}_df{}_el{}_dk{}_pk{}_ps{}_erp{}_rp{}_{}_{}'.format(
            args.model_id,
            args.d_model,
            args.d_ff,
            args.e_layers,
            args.dw_ks, 
            args.patch_ks,
            args.patch_sd,
            args.enable_res_param,
            args.re_param,
            args.des,ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mse,mae = exp.test(setting)
        mses.append(mse)
        maes.append(mae)

        # if args.do_predict:
        #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        #     exp.predict(setting, True)

        torch.cuda.empty_cache()
    print('average mse:{0:.3f}±{1:.3f}, mae:{2:.3f}±{3:.3f}'.format(np.mean(
    mses), np.std(mses), np.mean(maes), np.std(maes))) 
else:
    ii = 0
    setting = '{}_dm{}_df{}_el{}_dk{}_pk{}_ps{}_erp{}_rp{}_{}_{}'.format(
        args.model_id,
        args.d_model,
        args.d_ff,
        args.e_layers,
        args.dw_ks, 
        args.patch_ks,
        args.patch_sd,
        args.enable_res_param,
        args.re_param,
        args.des,ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()

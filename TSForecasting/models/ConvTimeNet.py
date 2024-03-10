__all__ = ['ConvTimeNet']

# Cell
import torch
from torch import nn
from layers.ConvTimeNet_backbone import ConvTimeNet_backbone

# ConvTimeNet: depatch + batch norm + gelu + Conv + 2-layer-ffn(PointWise Conv + PointWise Conv)
class Model(nn.Module):
    def __init__(self, configs, norm:str='batch', act:str="gelu", head_type = 'flatten'):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        head_dropout = configs.head_dropout
    
        patch_len = configs.patch_ks
        stride = configs.patch_sd
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        seq_len = configs.seq_len
        dw_ks = configs.dw_ks

        re_param = configs.re_param
        re_param_kernel = configs.re_param_kernel
        enable_res_param = configs.enable_res_param
    
        
        # model
        self.model = ConvTimeNet_backbone(c_in=c_in, seq_len=seq_len, context_window = context_window,
                                target_window=target_window, patch_len=patch_len, stride=stride, n_layers=n_layers, d_model=d_model, d_ff=d_ff, dw_ks=dw_ks, norm=norm, dropout=dropout, act=act,head_dropout=head_dropout, padding_patch = padding_patch, head_type=head_type, 
                                revin=revin, affine=affine, deformable=True, subtract_last=subtract_last, enable_res_param=enable_res_param, re_param=re_param, re_param_kernel=re_param_kernel)
        
    def forward(self, x):
        x = x.permute(0,2,1)    
        x = self.model(x)
        x = x.permute(0,2,1)   

        return x
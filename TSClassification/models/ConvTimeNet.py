import torch, math
import numpy as np
import torch.nn as nn

from models.dlutils import *

from models.ConvTimeNet_backbone import ConvTimeNet_backbone

class Model(nn.Module):
	def __init__(self, configs):
		super().__init__()
		self.name = 'Model'
		patch_size, patch_stride = configs.patch_size, configs.patch_stride
  
		# DePatch Embedding
		in_channel, out_channel, seq_len = configs.enc_in, configs.d_model, configs.seq_len
		self.depatchEmbedding = DeformablePatch(in_channel, out_channel, seq_len, patch_size, patch_stride)
  
		# ConvTimeNet Backbone
		new_len = self.depatchEmbedding.new_len
		c_in, c_out, dropout = out_channel, configs.num_class, configs.dropout,
		d_ff, d_model, dw_ks = configs.d_ff, configs.d_model, configs.dw_ks
  
		block_num, enable_res_param, re_param = len(dw_ks), True, True
		
		self.main_net = ConvTimeNet_backbone(c_in, c_out, new_len, block_num, d_model, d_ff, 
						  dropout, act='gelu', dw_ks=dw_ks, enable_res_param=enable_res_param, 
						  re_param=re_param, norm='batch', use_embed=False, device='cuda:0')

	def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
		out_patch = self.depatchEmbedding(x_enc) # [bs, ffn_output(before softmax)]
		output = self.main_net(out_patch.permute(0, 2, 1))	
		return output
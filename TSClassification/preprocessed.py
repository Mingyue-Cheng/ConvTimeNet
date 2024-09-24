import os, json, pytab
import numpy as np
import pandas as pd

from src.parsers import *
from scipy.io import arff
from pprint import pformat, pprint

arff_dataset = {
 'AWR': 'ArticularyWordRecognition', 
 'AF': 'AtrialFibrillation', 
 'BM': 'BasicMotions', 
 'CT': 'CharacterTrajectories', 
 'CR': 'Cricket', 
 'DD': 'DuckDuckGeese', 
 'EW': 'EigenWorms', 
 'EP': 'Epilepsy', 
 'ER': 'ERing', 
 'EC': 'EthanolConcentration', 
 'FD': 'FaceDetection', 
 'FM': 'FingerMovements', 
 'HMD': 'HandMovementDirection', 
 'HW': 'Handwriting',
 'HB': 'Heartbeat', 
 'IW': 'InsectWingbeat', 
 'JV': 'JapaneseVowels', 
 'LI': 'Libras', 
 'LSST': 'LSST', 
 'MI': 'MotorImagery', 
 'NATOPS': 'NATOPS', 
 'PEMS-SF': 'PEMS-SF', 
 'PD': 'PenDigits', 
 'PS': 'PhonemeSpectra', 
 'RS': 'RacketSports', 
 'SRS1': 'SelfRegulationSCP1', 
 'SRS2': 'SelfRegulationSCP2', 
 'SAD': 'SpokenArabicDigits', 
 'SW': 'StandWalkJump', 
 'UWG': 'UWaveGestureLibrary'
}

entities = {'HAR': ['har-0']}
for k in arff_dataset.keys():
	entities.update({k:[f'{k.lower()}-0']})

data_paths = {'HAR': 'dataset/UCI_HAR_Dataset'}
for k, v in arff_dataset.items():
	data_paths.update({k:f'dataset/{v}'})
	 
def normalize3(a, axis=None, min_a = None, max_a = None): # normalize to [-1, 1]
	if min_a is None:
		min_a, max_a = np.min(a, axis=axis, keepdims=True), np.max(a, axis=axis, keepdims=True)
		print('min_a: ', min_a.shape)
		print('max_a: ', max_a.shape)
  
	return 2. * (a - min_a) / (max_a - min_a + 0.0001) - 1., min_a, max_a

def padding_varying_length(data): # fill the NAN with 0 
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			data[i, j, :][np.isnan(data[i, j, :])] = 0
	return data

# def xlsx_to_csv():
#     workbook = xlrd.open_workbook('1.xlsx')
#     table = workbook.sheet_by_index(0)
#     with codecs.open('1.csv', 'w', encoding='utf-8') as f:
#         write = csv.writer(f)
#         for row_num in range(table.nrows):
#             row_value = table.row_values(row_num)
#             write.writerow(row_value)

def visulizing(train_d, test_d, dataset, file):
	import matplotlib.pyplot as plt
	plt.figure(figsize=(24, 7))
	
	import shutil
	if os.path.exists(f'visulization//datasets//train_{dataset}//{file}'):
		shutil.rmtree(f'visulization//datasets//train_{dataset}//{file}')
	if os.path.exists(f'visulization//datasets//test_{dataset}//{file}'):
		shutil.rmtree(f'visulization//datasets//test_{dataset}//{file}')

	os.makedirs(f'visulization//datasets//train_{dataset}//{file}', exist_ok=True)
	for k in range(train_d.shape[1]):
		plt.clf()
		plt.plot(train_d[:, k])

		plt.title(f'colume{k}')
		plt.savefig(f'visulization//datasets//train_{dataset}//{file}//{k}.png')
	os.makedirs(f'visulization//datasets//test_{dataset}//{file}', exist_ok=True)
	for k in range(test_d.shape[1]):
	
		plt.clf()
		plt.plot(test_d[:, k])

		plt.title(f'colume{k}')
		plt.savefig(f'visulization//datasets//test_{dataset}//{file}//{k}.png')
  
def draw_table(data_dict, save_dir):
	"""
	draw the table

	:param data: data format {
		'modelname': [model1, model2, ...],
		'f1': [f1_score1, f1_score2, ...], 
		...
		}
	:return:
	"""
	data = {'Dataset': [], 'Train Size': [], 'Test Size': [], 'Dimension': [], 'Length': [], 'Classes': []}
	for k, v in data_dict.items():
		data['Dataset'].append(k)
		for k1, v1 in v.items():
			data[k1].append(v1)
	
	# font to Chinese
	# pytab.plt.rcParams["font.sans-serif"] = ["SimHei"]
	
	pytab.table( 
		data=data,
		data_loc='center',
		# th_type='dark',
		th_c='#aaaaee',  # background color of the table header
		td_c='gray',  # background color of the table row
		table_type='striped',
		figsize=(len(data.keys()), int(len(data.values()) / len(data.keys()) + 1)),
		fontsize=30,
	)
	
	# pytab.show()
	pytab.save(save_dir)
	return 
  
def print_and_check_data(train_data, test_data, train_labels, test_labels):
	# print('train_data: ', train_data.shape)
	# print('test_data: ', test_data.shape)
	# print('train_labels: ', train_labels.shape)
	# print('test_labels: ', test_labels.shape)
	# print('train_l_class: ', np.unique(train_labels))
	# print('test_l_class: ', np.unique(test_labels))

	print('max/min in train: ', np.max(train_data), np.min(train_data))
	print('max/min in test: ', np.max(test_data), np.min(test_data))

	assert not np.isnan(train_data).any(), "Nan in train data!"
	assert not np.isnan(test_data).any(), "Nan in test data!"
  
def preprocess(dataset, file): # data format : (n_samples, n_feature)
	data_path = data_paths[dataset]
 
	save_dir = os.path.join('preprocessed', dataset, file)
	os.makedirs(save_dir, exist_ok=True)

	train_data, test_data = None, None
	train_labels, test_labels = None, None
 
	f1 = open("preprocessed/dataDescription.txt",'r+')
	dic = f1.read()
	dic = eval(dic) if dic != '' else {}
 
	if dataset in dic.keys():
		print(f"The {dataset} has been preprocessed!")
		return

	if dataset == 'HAR':
		train_dict, test_dict = {}, {}
		for axis in ['x', 'y', 'z']:
			train_dict[f'tot_acc_{axis}'] =  np.loadtxt(f'{data_path}/train/Inertial Signals/total_acc_{axis}_train.txt')
			test_dict[f'tot_acc_{axis}'] =  np.loadtxt(f'{data_path}/test/Inertial Signals/total_acc_{axis}_test.txt')
			# print(train_tot_acc[axis].shape) 

		for axis in ['x', 'y', 'z']:
			train_dict[f'body_acc_{axis}'] =  np.loadtxt(f'{data_path}/train/Inertial Signals/body_acc_{axis}_train.txt')
			test_dict[f'body_acc_{axis}'] =  np.loadtxt(f'{data_path}/test/Inertial Signals/body_acc_{axis}_test.txt')
			# print(train_body_acc[axis].shape)
			
		for axis in ['x', 'y', 'z']:
			train_dict[f'body_gyro_{axis}'] =  np.loadtxt(f'{data_path}/train/Inertial Signals/body_gyro_{axis}_train.txt')
			test_dict[f'body_gyro_{axis}'] =  np.loadtxt(f'{data_path}/test/Inertial Signals/body_gyro_{axis}_test.txt')
			# print(train_body_gyro[axis].shape)

		train_list, test_list = [], []
		for item in train_dict.values():
			train_list.append(np.expand_dims(item, axis=1))
		for item in test_dict.values():
			test_list.append(np.expand_dims(item, axis=1))

		train_data = np.concatenate(train_list, axis=1).transpose(0, 2, 1) # (numbers, window, channels)
		test_data = np.concatenate(test_list, axis=1).transpose(0, 2, 1) # (numbers, window, channels)

		train_data, min_a, max_a = normalize3(train_data, axis=(0,1))
		test_data, _, _ = normalize3(test_data, min_a=min_a, max_a=max_a)

		train_labels = np.loadtxt(f'{data_path}/train/y_train.txt')
		test_labels = np.loadtxt(f'{data_path}/test/y_test.txt')
  
		label_class = np.unique(train_labels)
		for k, label in enumerate(label_class):
			train_labels[np.where(train_labels==label)[0]] = k
			test_labels[np.where(test_labels==label)[0]] = k
   
		print_and_check_data(train_data, test_data, train_labels, test_labels)
  
		data = {'train_d': train_data, 'test_d': test_data, \
	  			'train_l': train_labels, 'test_l': test_labels}
  
		for name in data.keys():
			np.save(os.path.join(save_dir, f'{name}.npy'), data[name])
  
	elif dataset in arff_dataset.keys():
		# from Formertime
		train_path = f'{data_path}//{arff_dataset[dataset]}_TRAIN.arff'
		test_path = f'{data_path}//{arff_dataset[dataset]}_TEST.arff'
  
		### load training data
		train_data, train_labels = [], []
		label_dict = {}
		label_index = 0
		with open(train_path, encoding='UTF-8', errors='ignore') as f:
			data, meta = arff.loadarff(f)
			f.close()
   
		assert type(data[0][0]) == np.ndarray  # multivariate
	  
		for index in range(data.shape[0]):
			raw_data = data[index][0]
			raw_label = data[index][1]
			if label_dict.__contains__(raw_label):
				train_labels.append(label_dict[raw_label])
			else:
				label_dict[raw_label] = label_index
				train_labels.append(label_index)
				label_index += 1
			raw_data_list = raw_data.tolist()
			# print(raw_data_list)
			train_data.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))
		
		### load testing data
		test_data, test_labels = [], []
		with open(test_path, encoding='UTF-8', errors='ignore') as f:
			data, meta = arff.loadarff(f)
			f.close()
   
		for index in range(data.shape[0]):
			raw_data = data[index][0]
			raw_label = data[index][1]
			test_labels.append(label_dict[raw_label])
			raw_data_list = raw_data.tolist()
			test_data.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

		train_data = padding_varying_length(np.array(train_data))
		test_data = padding_varying_length(np.array(test_data))

		train_data, test_data = np.array(train_data), np.array(test_data)
		train_labels, test_labels = np.array(train_labels), np.array(test_labels)
  
		train_data, min_a, max_a = normalize3(train_data, axis=(0,1))
		test_data, _, _ = normalize3(test_data, min_a=min_a, max_a=max_a)
  
		print_and_check_data(train_data, test_data, train_labels, test_labels)
  
		# if dataset == 'CT':
		# 	visulizing(train_data[0], test_data[0], dataset, file)
  
		data = {'train_d': train_data, 'test_d': test_data, \
	  			'train_l': train_labels, 'test_l': test_labels}
  
		for name in data.keys():
			np.save(os.path.join(save_dir, f'{name}.npy'), data[name])
  
	dp_dict = {
		'Train Size': train_data.shape[0],
		'Test Size': test_data.shape[0],
		'Length': train_data.shape[1],
		'Dimension': train_data.shape[2],
		'Classes': len(list(set(train_labels)))
	}
	dic.update({dataset:dp_dict})
	draw_table(dic, save_dir='preprocessed//dataDescription.png')
 
	f1.seek(0); f1.truncate()
	f1.write(pformat(dic)); f1.close()

def check_all_dataset():
	f1 = open("preprocessed/dataDescription.txt",'r+')
	dic = eval(f1.read())
	list_2 = [2**k for k in range(12)]
	def find_2_pow(number):
		for k in list_2:
			if number < k: return k 
				
 
	tgt_idx, tgt_idx_dict = 'Dimension', {}
	for k, v in dic.items():
		d = v[tgt_idx]	
		pow_2 = find_2_pow(d)
  
		tgt_idx_dict.update({k:v[tgt_idx]})
		# tgt_idx_dict.update({k:max(8, pow_2)})
	pprint(tgt_idx_dict)
	exit(0)

if __name__ == '__main__':
	check_all_dataset()
	datasets = ['FD']
	
	for dataset in datasets:
		file_list = entities[dataset]
		for file in file_list:
			print(f'preprocessing {dataset} {file}...')
			preprocess(dataset, file)
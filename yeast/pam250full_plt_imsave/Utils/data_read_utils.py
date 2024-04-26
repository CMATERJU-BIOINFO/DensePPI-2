import scipy.io
import json, pickle, os, math, shutil
import pandas as pd
import numpy as np
from .image_utils import SIZE, maxAllowedSequenceLength, checkChars

def clean_dataframes(pos_df : pd.DataFrame, neg_df : pd.DataFrame, make_dfs_equal = True):
	print("Before cleaning .....")
	print("Positive Interactions : ", len(pos_df))
	print("Negative Interactions : ", len(neg_df))
	
	mask = lambda x : checkChars(x['Protein_seq1'], x['Protein_seq2'])
	pos_df = pos_df.loc[pos_df.apply(mask, axis = 1)]
	neg_df = neg_df.loc[neg_df.apply(mask, axis = 1)]

	mask = (pos_df['Protein_seq1'].str.len() >= SIZE) & (pos_df['Protein_seq2'].str.len() >= SIZE) & (pos_df['Protein_seq1'].str.len() < maxAllowedSequenceLength) & (pos_df['Protein_seq2'].str.len() < maxAllowedSequenceLength)
	pos_df = pos_df.loc[mask]
	mask = (neg_df['Protein_seq1'].str.len() >= SIZE) & (neg_df['Protein_seq2'].str.len() >= SIZE) & (neg_df['Protein_seq1'].str.len() < maxAllowedSequenceLength) & (neg_df['Protein_seq2'].str.len() < maxAllowedSequenceLength) 
	neg_df = neg_df.loc[mask]
	
	if make_dfs_equal :
		minRows = min(len(pos_df), len(neg_df))
		if (minRows == len(pos_df)):
			neg_df = neg_df.sample(n = minRows)
		else:
			pos_df = pos_df.sample(n = minRows)
		
	print("After cleaning .....")
	print("Positive Interactions : ", len(pos_df))
	print("Negative Interactions : ", len(neg_df))
	
	pos_df.reset_index(drop=True, inplace=True)
	neg_df.reset_index(drop=True, inplace=True)

	return pos_df, neg_df

def createDF_PanEtAl(lines : list):
	del lines[:4]
	df = pd.DataFrame(columns=['Index', 'Protein_1', 'Protein_2', 
				'Protein_seq1', 'Protein_seq2'])
	
	i = 0
	count = 1

	while (i < len(lines)):
		li = list()
		li.append(count)
		li.extend(lines[i][:-1].split()[1:]) #removing '\n'
		li.append(lines[i+2][:-1])
		li.append(lines[i+4][:-1]) 
		df = df._append(pd.Series(li, index=df.columns), ignore_index=True)
		count+=1
		i+=5

	return df

def parse_images_from_txt_PanEtAl(pos_loc : str, neg_loc : str, out_dir: str, make_dfs_eq = True):
	with open(pos_loc) as f:
		lines = f.readlines()
	f.close()

	pos_df = createDF_PanEtAl(lines)
	pout_loc = out_dir + 'positive.pkl'

	with open(neg_loc) as f:
		lines = f.readlines()
	f.close()
		
	neg_df = createDF_PanEtAl(lines)
	nout_loc = out_dir + 'negative.pkl'

	df1, df2 = clean_dataframes(pos_df, neg_df, make_dfs_eq)

	df1.to_pickle(pout_loc)
	df2.to_pickle(nout_loc)

def createDF_sCerevisiae(P1 : list, P2 : list, P1_prefix : str, P2_prefix : str):
	df = pd.DataFrame(columns=['Index', 'Protein_1', 'Protein_2', 
				'Protein_seq1', 'Protein_seq2'])
	count = 1
	for i in range(len(P1)):
		li = list()
		li.append(count)
		li.append(P1_prefix + str(count))
		li.append(P2_prefix + str(count))
		li.append(P1[i])
		li.append(P2[i]) 
		df = df._append(pd.Series(li, index=df.columns), ignore_index=True)
		count+=1
	
	return df

def copy_pos_neg_pkl_from_prev_parsing(pkl_in_loc : str, out_loc : str):
    pos_pkl_loc = os.path.join(pkl_in_loc, 'positive.pkl')
    neg_pkl_loc = os.path.join(pkl_in_loc, 'negative.pkl')
    
    shutil.copy(pos_pkl_loc, out_loc)
    shutil.copy(neg_pkl_loc, out_loc)

def parse_sCerevisiae_data(NAmat_loc : str, NBmat_loc : str, PAmat_loc : str, PBmat_loc : str, out_dir : str, make_dfs_eq = True):
	NAmat = scipy.io.loadmat(NAmat_loc)
	NBmat = scipy.io.loadmat(NBmat_loc)
	PAmat = scipy.io.loadmat(PAmat_loc)
	PBmat = scipy.io.loadmat(PBmat_loc)

	P1 = [PAmat['P_protein_A'][i, 0][0] for i in range(len(PAmat['P_protein_A']))]
	P2 = [PBmat['P_protein_B'][i, 0][0] for i in range(len(PBmat['P_protein_B']))]
	N1 = [NAmat['N_protein_A'][i, 0][0] for i in range(len(NAmat['N_protein_A']))]
	N2 = [NBmat['N_protein_B'][i, 0][0] for i in range(len(NBmat['N_protein_B']))]

	pos_df = createDF_sCerevisiae(P1, P2, "SCP1_", "SCP2_")
	pout_loc = out_dir + 'positive.pkl'

	neg_df = createDF_sCerevisiae(N1, N2, "SCN1_", "SCN2_")
	nout_loc = out_dir + 'negative.pkl'

	df1, df2 = clean_dataframes(pos_df, neg_df, make_dfs_eq)

	df1.to_pickle(pout_loc)
	df2.to_pickle(nout_loc)

def process_and_return_dirs(out_dir : str):
	min_data_frac = 0.2
	orig_img_pos_out_pkl_loc = out_dir + 'positive.pkl'
	orig_img_neg_out_pkl_loc = out_dir + 'negative.pkl'
	POSL=os.listdir("{}positive/".format(out_dir))
	NEGL=os.listdir("{}negative/".format(out_dir))
	print("Total: len(POS) = {}, len(NEG) = {}".format(len(POSL),len(NEGL)))
	out_loc = out_dir + 'data_train_test.pkl'
	with open(out_loc, 'rb') as file:
		myvar = pickle.load(file)
	pos_train_img_list = myvar[0]
	neg_train_img_list = myvar[1]
	pos_test_img_list = myvar[2]
	neg_test_img_list = myvar[3]
	print("Total: len(POS_train) = {}, len(NEG_train) = {}".format(len(pos_train_img_list), len(neg_train_img_list)))
	print("Total: len(POS_test) = {}, len(NEG_test) = {}".format(len(pos_test_img_list), len(neg_test_img_list)))
	
	pos_train_sub_img_loc = out_dir + "train" + "/complete/pos_sub/"
	pos_train_sub_img_dict = out_dir + "positive" + "_" + "train" + "_image_dict.json"
	neg_train_sub_img_loc = out_dir + "train" + "/complete/neg_sub/"
	neg_train_sub_img_dict = out_dir + "negative" + "_" + "train" + "_image_dict.json"
	pos_test_sub_img_loc = out_dir + "test" + "/complete/pos_sub/"
	pos_test_sub_img_dict = out_dir + "positive" + "_" + "test" + "_image_dict.json"
	neg_test_sub_img_dict = out_dir + "negative" + "_" + "test" + "_image_dict.json"
	
	f = open(pos_train_sub_img_dict)
	pimage_dict = json.load(f)
	f.close()

	f = open(neg_train_sub_img_dict)
	nimage_dict = json.load(f)
	f.close()

	print("Positive dictionary size : ", len(pimage_dict.keys()))
	print("Negative dictionary size : ", len(nimage_dict.keys()))
	pos_sub_images = [len(pimage_dict[k]) for k in pimage_dict.keys()]
	neg_sub_images = [len(nimage_dict[k]) for k in nimage_dict.keys()]
	print("Positive Sub-image Count from dictionary :", sum(pos_sub_images))
	print("Positive Sub-image Count from dictionary :", sum(neg_sub_images))
	searchScope = int(max(np.std(pos_sub_images), np.std(neg_sub_images)))
	minSubImageCount = int(min_data_frac * min(sum(pos_sub_images), sum(neg_sub_images)))
	print("Choosing Minimum SubImages as : ", minSubImageCount)
	print("Choosing searchSCope : ", searchScope)

	minDiff = math.inf
	final_threshold = 0

	for thresh in range(0, searchScope + 1, 1):
		pli = [len(pimage_dict[k]) for k in pimage_dict.keys() if len(pimage_dict[k]) <= thresh]
		nli = [len(nimage_dict[k]) for k in nimage_dict.keys() if len(nimage_dict[k]) <= thresh]
		if ((minDiff > abs(sum(pli) - sum(nli))) and (sum(pli) > minSubImageCount) and (sum(nli) > minSubImageCount)):
			minDiff = abs(sum(pli) - sum(nli))
			final_threshold = thresh

	image_keys_p = [k for k in pimage_dict.keys() if len(pimage_dict[k]) <= final_threshold]
	image_keys_n = [k for k in nimage_dict.keys() if len(nimage_dict[k]) <= final_threshold]
	sub_p = [pimage_dict[k] for k in pimage_dict.keys() if len(pimage_dict[k]) <= final_threshold]
	sub_n = [nimage_dict[k] for k in nimage_dict.keys() if len(nimage_dict[k]) <= final_threshold]

	sub_p = [item for sublist in sub_p for item in sublist]
	sub_n = [item for sublist in sub_n for item in sublist]
	
	equal_train_data_pkl_loc = out_dir + "equal_train_data.pkl"

	with open(equal_train_data_pkl_loc, 'rb') as file:
		myvar = pickle.load(file)
	image_keys_p = myvar[0]
	image_keys_n = myvar[1]
	sub_p = myvar[2]
	sub_n = myvar[3]
	print("Total images selected in positive train after Equalization : ", len(image_keys_p))
	print("Total images selected in negative train after Equalization : ", len(image_keys_n))
	print("Total sub-images in positive train after Equalization : ", len(sub_p))
	print("Total sub-images in negative train after Equalization : ", len(sub_n))

	eq_neg_train_sub_img_loc = out_dir + "/train/equal/neg_sub/"

	return pos_train_sub_img_loc, pos_train_sub_img_dict, neg_train_sub_img_loc, neg_train_sub_img_dict, pos_test_sub_img_loc, pos_test_sub_img_dict, neg_test_sub_img_dict, equal_train_data_pkl_loc, eq_neg_train_sub_img_loc
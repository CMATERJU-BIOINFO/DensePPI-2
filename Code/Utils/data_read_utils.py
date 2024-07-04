import scipy.io
import json, pickle, os, math, shutil, copy
import pandas as pd
import numpy as np
from Bio import SeqIO
import networkx as nx
from .image_utils import SIZE, STRIDE, maxAllowedSequenceLength, checkChars, generate_images, Image
from .path_utils import createLocationIfNotExists

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

def createDF_sCerevisiae(P1 : list, P2 : list, P_prefix : str):
	df = pd.DataFrame(columns=['Index', 'Protein_1', 'Protein_2', 
				'Protein_seq1', 'Protein_seq2'])
	total_P = copy.deepcopy(P1)
	total_P.extend(copy.deepcopy(P2))
	P_unique = np.unique(np.array(total_P))
	P_name_list = [P_prefix + str(i + 1) for i in range(P_unique.shape[0])]
	P_dict = {key : val for (key, val) in zip(P_unique, P_name_list)}
	count = 1
	for i in range(len(P1)):
		li = list()
		li.append(count)
		li.append(P_dict[P1[i]])
		li.append(P_dict[P2[i]])
		li.append(P1[i])
		li.append(P2[i]) 
		df = df._append(pd.Series(li, index=df.columns), ignore_index=True)
		count+=1
	
	return df

def copy_pos_neg_pkl_from_prev_parsing(pkl_in_loc : str, out_loc : str):
	createLocationIfNotExists(out_loc)
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

	pos_df = createDF_sCerevisiae(P1, P2, "SCP_")
	pout_loc = out_dir + 'positive.pkl'

	neg_df = createDF_sCerevisiae(N1, N2, "SCN_")
	nout_loc = out_dir + 'negative.pkl'

	df1, df2 = clean_dataframes(pos_df, neg_df, make_dfs_eq)

	df1.to_pickle(pout_loc)
	df2.to_pickle(nout_loc)

def createDF_edgeList_seq(pandas_edge_list : pd.DataFrame, seq : dict):
	df = pd.DataFrame(columns=['Index', 'Protein_1', 'Protein_2', 
				'Protein_seq1', 'Protein_seq2'])

	i = 0
	count = 1

	while (i < len(pandas_edge_list)):
		P1 = pandas_edge_list.iloc[i, 0]
		P2 = pandas_edge_list.iloc[i, 1]
		li = list()
		li.append(count)
		li.append(P1)
		li.append(P2)
		li.append(seq[P1]) 
		li.append(seq[P2])
		df = df._append(pd.Series(li, index=df.columns), ignore_index=True)
		count+=1
		i+=1

	return df

def parse_images_from_edgeList_seq(pandas_edge_list : pd.DataFrame, seq : dict, out_loc: str):
	df = createDF_edgeList_seq(pandas_edge_list, seq)
	print("Before cleaning .....")
	print("Positive Interactions : ", len(df))

	mask = lambda x : checkChars(x['Protein_seq1'], x['Protein_seq2'])
	df = df.loc[df.apply(mask, axis = 1)]

	mask = (df['Protein_seq1'].str.len() >= SIZE) & (df['Protein_seq2'].str.len() >= SIZE) & (df['Protein_seq1'].str.len() < maxAllowedSequenceLength) & (df['Protein_seq2'].str.len() < maxAllowedSequenceLength)
	df = df.loc[mask]

	print("After cleaning .....")
	print("Positive Interactions : ", len(df))

	df.reset_index(drop=True, inplace=True)
	df.to_pickle(out_loc)
	return df
 
def id_to_seq_dict_from_fasta(loc : str):
	fasta_sequences = SeqIO.parse(open(loc),'fasta')
	fasta_id_to_seq_dict = {}
	for fasta in fasta_sequences:
		name, sequence = fasta.id, str(fasta.seq)
		fasta_id_to_seq_dict[name] = sequence

	return fasta_id_to_seq_dict
 
def create_df_from_pd_df(in_df : pd.DataFrame, id_to_seq : dict):
	df = pd.DataFrame(columns=['Index', 'Protein_1', 'Protein_2', 
						   'Protein_seq1', 'Protein_seq2'])
	count = 1
	for i in range(len(in_df)):
		li = list()
		li.append(count)
		p1 = in_df.iloc[i]['IntA']
		p2 = in_df.iloc[i]['IntB']
		li.append(p1)
		li.append(p2)
		try:
			li.append(id_to_seq[p1])
			li.append(id_to_seq[p2]) 
		except KeyError:
			if ((i - count) > 1000):
				print("More than 1000 pais skipped !!! Something Wrong with sequence file !!!")
				print("Showing last info :")
				print("df_idx = {}, p1 = {}, p2 = {}".format(str(i), p1, p2))
				print("Exiting...")
				
				exit(0)
			continue
		df = df._append(pd.Series(li, index=df.columns), ignore_index=True)
		count+=1
	print("Discarded sequences count : ", len(in_df) - len(df))
		
	return df

def create_train_test_imgs_and_return_split(train_csv_loc : str, test_csv_loc : str, fasta_seq_loc : str, out_dir : str):
	id_to_seq_dict = id_to_seq_dict_from_fasta(fasta_seq_loc)
	
	print("Creating original training images.....")
	train_df = pd.read_csv(train_csv_loc)
	train_base_dir = "{}/train/".format(out_dir)
	pos_train_data = train_df[train_df['Class'] == 1].reset_index(drop=True) # use 'Cls' for C1-fold1
	neg_train_data = train_df[train_df['Class'] == 0].reset_index(drop=True) # use 'Class' for C2-fold1 and C3-fold1
	pos_train_df_save_loc = "{}/pos_train.pkl".format(out_dir)
	neg_train_df_save_loc = "{}/neg_train.pkl".format(out_dir)
	pos_train_df = create_df_from_pd_df(pos_train_data, id_to_seq_dict)
	neg_train_df = create_df_from_pd_df(neg_train_data, id_to_seq_dict)
	df1, df2 = clean_dataframes(pos_train_df, neg_train_df, False)
	df1.to_pickle(pos_train_df_save_loc)
	df2.to_pickle(neg_train_df_save_loc)
	generate_images(pos_train_df_save_loc, 'positive', train_base_dir)
	generate_images(neg_train_df_save_loc, 'negative', train_base_dir)
	print("Creating original training images finished.....")
	
	print("Creating original testing images.....")
	test_df = pd.read_csv(test_csv_loc)
	test_base_dir = "{}/test/".format(out_dir)
	pos_test_data = test_df[test_df['Class'] == 1].reset_index(drop=True) # use 'Cls' for C1-fold1
	neg_test_data = test_df[test_df['Class'] == 0].reset_index(drop=True) # use 'Class' for C2-fold1 and C3-fold1
	pos_test_df_save_loc = "{}/pos_test.pkl".format(out_dir)
	neg_test_df_save_loc = "{}/neg_test.pkl".format(out_dir)
	pos_test_df = create_df_from_pd_df(pos_test_data, id_to_seq_dict)
	neg_test_df = create_df_from_pd_df(neg_test_data, id_to_seq_dict)
	df1, df2 = clean_dataframes(pos_test_df, neg_test_df, False)
	df1.to_pickle(pos_test_df_save_loc)
	df2.to_pickle(neg_test_df_save_loc)
	generate_images(pos_test_df_save_loc, 'positive', test_base_dir)
	generate_images(neg_test_df_save_loc, 'negative', test_base_dir)
	print("Creating original testing images finished.....")
 
	pos_train_imgl = os.listdir("{}/positive/".format(train_base_dir))
	neg_train_imgl = os.listdir("{}/negative/".format(train_base_dir))
	pos_test_imgl = os.listdir("{}/positive/".format(test_base_dir))
	neg_test_imgl = os.listdir("{}/negative/".format(test_base_dir))
	
	return train_base_dir, test_base_dir, pos_train_imgl, neg_train_imgl, pos_test_imgl, neg_test_imgl
	
def process_and_return_dirs(out_dir : str, is_equal_split = True):
	min_data_frac = 0.2

	print("===STATS FOR ORIGINAL AND COMPLETE TRAINING & TEST IMAGES===")
	pos_train_sub_img_loc = out_dir + "train" + "/complete/pos_sub/"
	pos_train_sub_img_dict = out_dir + "positive" + "_" + "train" + "_image_dict.json"
	neg_train_sub_img_loc = out_dir + "train" + "/complete/neg_sub/"
	neg_train_sub_img_dict = out_dir + "negative" + "_" + "train" + "_image_dict.json"
	pos_test_sub_img_loc = out_dir + "test" + "/complete/pos_sub/"
	pos_test_sub_img_dict = out_dir + "positive" + "_" + "test" + "_image_dict.json"
	neg_test_sub_img_dict = out_dir + "negative" + "_" + "test" + "_image_dict.json"
	

	print("===COUNTS READING PICKLE & JSON FILE===")
	original_img_split_pkl_loc = out_dir + 'data_train_test.pkl'
	with open(original_img_split_pkl_loc, 'rb') as file:
		myvar = pickle.load(file)
	pos_train_img_list = myvar[0]
	neg_train_img_list = myvar[1]
	pos_test_img_list = myvar[2]
	neg_test_img_list = myvar[3]
	print("Total orig: len(POS_train) = {}, len(NEG_train) = {}".format(len(pos_train_img_list), len(neg_train_img_list)))
	print("Total orig: len(POS_test) = {}, len(NEG_test) = {}".format(len(pos_test_img_list), len(neg_test_img_list)))
	
	eq_neg_train_sub_img_loc = None
	equal_train_data_pkl_loc = None
	if is_equal_split:
		eq_neg_train_sub_img_loc = out_dir + "/train/equal/neg_sub/"
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
	print("Negative Sub-image Count from dictionary :", sum(neg_sub_images))
	
	if is_equal_split:
		searchScope = int(max(np.std(pos_sub_images), np.std(neg_sub_images)))
		minSubImageCount = int(min_data_frac * min(sum(pos_sub_images), sum(neg_sub_images)))

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
 
		print("===COUNTS USING CALCULATION===")
		print("Choosing Minimum SubImages as : ", minSubImageCount)
		print("Choosing searchSCope : ", searchScope)
		print("Total images selected in positive train after Equalization : ", len(image_keys_p))
		print("Total images selected in negative train after Equalization : ", len(image_keys_n))
		print("Total sub-images in positive train after Equalization : ", len(sub_p))
		print("Total sub-images in negative train after Equalization : ", len(sub_n))
		print("Equalized with threshold : ", final_threshold)
 
	print("===COUNTS READING FILE LOCATIONS===")
	POSL_complete=os.listdir("{}positive/".format(out_dir))
	NEGL_complete=os.listdir("{}negative/".format(out_dir))
	print("Total orig : len(POS) = {}, len(NEG) = {}".format(len(POSL_complete),len(NEGL_complete)))
	POSL_complete_sub=os.listdir(pos_train_sub_img_loc)
	NEGL_complete_sub=os.listdir(neg_train_sub_img_loc)
	print("Total complete_sub : len(POS) = {}, len(NEG) = {}".format(len(POSL_complete_sub),len(NEGL_complete_sub)))
	if is_equal_split:
		POSL_equal_sub=os.listdir(out_dir + "/train/equal/pos_sub/")
		NEGL_equal_sub=os.listdir(eq_neg_train_sub_img_loc)
		print("Total equal_sub : len(POS) = {}, len(NEG) = {}".format(len(POSL_equal_sub),len(NEGL_equal_sub)))

	return pos_train_sub_img_loc, pos_train_sub_img_dict, neg_train_sub_img_loc, neg_train_sub_img_dict, pos_test_sub_img_loc, pos_test_sub_img_dict, neg_test_sub_img_dict, equal_train_data_pkl_loc, eq_neg_train_sub_img_loc

def get_pandas_edge_list_from_json_paths(out_path_ids : str, out_path_p1_p2 : str) :
	f = open (out_path_ids, "r")
	p_data = json.load(f)
	f.close()

	pA = []
	pB = []
	pgr = nx.Graph()
	for i in range(len(p_data)):
		pA.append(p_data[i]['pA'])
		pB.append(p_data[i]['pB'])
		pgr.add_edge(p_data[i]['pA'], p_data[i]['pB'])

	f = open (out_path_p1_p2, "r")
	seq_dict = json.load(f)
	f.close()

	return nx.to_pandas_edgelist(pgr), seq_dict

def generate_test_results_to_csv(test_data_split_pkl_loc : str, test_data_results_loc : str, orig_img_pkl_base_loc : str, output_csv_loc : str) :
	with open(test_data_split_pkl_loc, 'rb') as file:
		myvar = pickle.load(file)

	pos_test_img_list = myvar[2]
	neg_test_img_list = myvar[3]

	f = open(test_data_results_loc + "positive_prediction_result_dict.json")
	p_data = json.load(f)
	f.close()

	f = open(test_data_results_loc + "negative_prediction_result_dict.json")
	n_data = json.load(f)
	f.close()

	df = pd.read_pickle(orig_img_pkl_base_loc + "negative.pkl")
	args = []
	for it in range(len(df)):
		args.append(df.iloc[it, 1] + '_' + df.iloc[it, 2] + '.png')

	args = np.array(args)
	out_df = pd.DataFrame(columns=['Index', 'Protein_1', 'Protein_2', 
            'Protein_seq1', 'Protein_seq2', 'img_dim', 'resultant_scores'])

	for i in range(len(n_data)):
		protein = neg_test_img_list[i]
		im = Image.open(os.path.dirname(output_csv_loc) + '/negative/' + protein)
		im = np.array(im)
		height, width, _ = im.shape
		imgDim = [len(list(range(0, height-SIZE+STRIDE, STRIDE))), len(list(range(0, width-SIZE+STRIDE, STRIDE)))]
		str_to_end_with = protein.split('_', 1)[1]
		idx = np.where(args == str_to_end_with)[0][0]
		new_row = {
			'Index' : i+1,
			'Protein_1' : df.iloc[idx][1],
			'Protein_2' : df.iloc[idx][2],
			'Protein_seq1' : df.iloc[idx][3],
			'Protein_seq2' : df.iloc[idx][4],
			'img_dim' : imgDim,
			'resultant_scores' : n_data[str(i + 1)]
		}
		out_df = out_df._append(pd.Series(new_row), ignore_index=True)
	
	out_df.to_csv(output_csv_loc + 'negative_result.csv', index=False)

	df = pd.read_pickle(orig_img_pkl_base_loc + "positive.pkl")
	args = []
	for it in range(len(df)):
		args.append(df.iloc[it, 1] + '_' + df.iloc[it, 2] + '.png')

	args = np.array(args)
	out_df = pd.DataFrame(columns=['Index', 'Protein_1', 'Protein_2', 
            'Protein_seq1', 'Protein_seq2', 'img_dim', 'resultant_scores'])

	for i in range(len(p_data)):
		protein = pos_test_img_list[i]
		im = Image.open(os.path.dirname(output_csv_loc) + '/positive/' + protein)
		im = np.array(im)
		height, width, _ = im.shape
		imgDim = [len(list(range(0, height-SIZE+STRIDE, STRIDE))), len(list(range(0, width-SIZE+STRIDE, STRIDE)))]
		str_to_end_with = protein.split('_', 1)[1]
		idx = np.where(args == str_to_end_with)[0][0]
		new_row = {
			'Index' : i+1,
			'Protein_1' : df.iloc[idx][1],
			'Protein_2' : df.iloc[idx][2],
			'Protein_seq1' : df.iloc[idx][3],
			'Protein_seq2' : df.iloc[idx][4],
			'img_dim' : imgDim,
			'resultant_scores' : p_data[str(i + 1)]
		}
		out_df = out_df._append(pd.Series(new_row), ignore_index=True)
	
	out_df.to_csv(output_csv_loc + 'positive_result.csv', index=False)


def generate_test_results_to_csv_single_type(test_data_split_pkl_loc : str, pos_test_data_results_loc : str, orig_pos_img_pkl_base_loc : str, output_csv_loc : str) :
	with open(test_data_split_pkl_loc, 'rb') as file:
		myvar = pickle.load(file)

	pos_test_img_list = myvar[2]

	f = open(pos_test_data_results_loc)
	p_data = json.load(f)
	f.close()

	df = pd.read_pickle(orig_pos_img_pkl_base_loc)
	args = []
	for it in range(len(df)):
		args.append(df.iloc[it, 1] + '_' + df.iloc[it, 2] + '.png')

	args = np.array(args)
	out_df = pd.DataFrame(columns=['Index', 'Protein_1', 'Protein_2', 
				'Protein_seq1', 'Protein_seq2', 'img_dim', 'resultant_scores'])

	for i in range(len(p_data)):
		protein = pos_test_img_list[i]
		im = Image.open(os.path.dirname(output_csv_loc) + '/positive/' + protein)
		im = np.array(im)
		height, width, _ = im.shape
		imgDim = [len(list(range(0, height-SIZE+STRIDE, STRIDE))), len(list(range(0, width-SIZE+STRIDE, STRIDE)))]
		str_to_end_with = protein.split('_', 1)[1]
		idx = np.where(args == str_to_end_with)[0][0]
		new_row = {
			'Index' : i+1,
			'Protein_1' : df.iloc[idx][1],
			'Protein_2' : df.iloc[idx][2],
			'Protein_seq1' : df.iloc[idx][3],
			'Protein_seq2' : df.iloc[idx][4],
			'img_dim' : imgDim,
			'resultant_scores' : p_data[str(i + 1)]
		}
		out_df = out_df._append(pd.Series(new_row), ignore_index=True)
	
	out_df.to_csv(output_csv_loc + 'positive_result.csv', index=False)

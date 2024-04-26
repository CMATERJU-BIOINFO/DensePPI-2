import os, itertools, scipy.io, re
import urllib.request
from .path_utils import createLocationIfNotExists
from .image_utils import *
from train.model import *
import networkx as nx
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

cd_hit_command = ""

def parse_data_from_scipy_mat(mat_loc : str, out_loc : str, file_save_prefix : str, protein_prefix : str, num_rows = None):
	PAmat = scipy.io.loadmat(mat_loc)
	PBmat = PAmat.copy()
	
	if num_rows == None:
		num_rows = len(PAmat['protein_A'])


	P1 = [PAmat['protein_A'][i, 0][0] for i in range(num_rows)]
	P2 = [PBmat['protein_B'][i, 0][0] for i in range(num_rows)]

	unique = list(dict.fromkeys(list(itertools.chain(P1, P2))))

	protdict = {}
	for i in range(len(unique)):
		protdict[unique[i]] = protein_prefix + str(i)

	PA = [protdict[prot] for prot in P1]
	PB = [protdict[prot] for prot in P2]

	pdict = [{'pA': pA, 'pB': pB} for pA, pB in zip(PA, PB)]

	protdict_map = {v: k for k, v in protdict.items()}

	out_path_ids = os.path.join(out_loc, "positive_" + file_save_prefix + "_IDs.json")
	out_path_p1_p2 = os.path.join(out_loc, "positive_" + file_save_prefix + "_P1_P2.json")

	with open(out_path_ids, 'w') as outfile:
		json.dump(pdict, outfile)   

	with open(out_path_p1_p2, 'w') as outfile:
		json.dump(protdict_map, outfile)

	return out_path_ids, out_path_p1_p2


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

def read_benchmark_data(benchmark_df_parent_loc : str, out_loc : str, prefix : str):
	out_path = os.path.join(out_loc, "pos_neg_" + prefix + "_IDs_to_seq.json")

	if (os.path.exists(out_path)):
		return out_path

	df_pos = pd.read_pickle(os.path.join(benchmark_df_parent_loc, "positive.pkl"))
	df_neg = pd.read_pickle(os.path.join(benchmark_df_parent_loc, "negative.pkl"))

	total_proteins_bench = set(df_neg['Protein_1']) | set(df_pos['Protein_1']) | set(df_neg['Protein_2']) | set(df_pos['Protein_2'])

	total_proteins_bench = list(total_proteins_bench)
	len(total_proteins_bench)

	new_df = df_neg[['Protein_1', 'Protein_seq1']]
	new_df.columns = ['ID', 'Sequence']
	temp = df_neg[['Protein_2', 'Protein_seq2']]
	temp.columns = ['ID', 'Sequence']
	new_df = new_df._append(temp, ignore_index = True)
	temp = df_pos[['Protein_1', 'Protein_seq1']]
	temp.columns = ['ID', 'Sequence']
	new_df = new_df._append(temp, ignore_index = True)
	temp = df_pos[['Protein_2', 'Protein_seq2']]
	temp.columns = ['ID', 'Sequence']
	new_df = new_df._append(temp, ignore_index = True)

	benchdict = new_df.set_index('ID')['Sequence'].to_dict()

	with open(out_path, 'w') as outfile:
		json.dump(benchdict, outfile)

	return out_path

def write_fasta_from_two_dicts(dict_1_path : str, dict_2_path : str, out_dir : str, prefix : str):
	f = open(dict_1_path, "r")
	dict_a = json.load(f)
	f.close()
	f = open(dict_2_path, "r")
	dict_b = json.load(f)
	f.close()
	res = {**dict_a, **dict_b}

	out_path_fasta = os.path.join(out_dir, "combined_" + prefix + ".fasta")
	ofile = open(out_path_fasta, "w")
	for protein in res.keys():
		ofile.write(">" + protein + "\n" + res[protein] + "\n")
	ofile.close()

	out_path_seq_to_ids = os.path.join(out_dir, "combined_" + prefix + "_IDs_to_seq.json")
	with open(out_path_seq_to_ids, 'w') as outfile:
		json.dump(res, outfile)

	return out_path_fasta

def read_fasta_out_clstrs(clstr_out_loc : str):
	file1 = open(clstr_out_loc, 'r')
	Lines = file1.readlines()
	LineIndex = 0
	cd_hit_dict = {}
	while LineIndex < len(Lines):
		FF = re.search(r'^(>Cluster.*)', Lines[LineIndex])
		if FF != None:
			LineIndex += 1
			li = []
			dkey = ''
			while LineIndex < len(Lines):
				KK = re.search(r'^(.*\*)', Lines[LineIndex])
				if KK != None:
					dkey = KK.group()
					dkey = re.sub(r'.*>','',dkey)
					dkey = re.sub(r'\.\.\. \*','',dkey)
				else:
					KK = re.search(r'^(.*\.\.\.)', Lines[LineIndex])
					dval = KK.group()
					dval = re.sub(r'.*>','',dval)
					dval = re.sub(r'\.\.\.','',dval)
					li.append(dval)
				if(LineIndex == len(Lines)-1):
					break
				FF = re.search(r'^(>Cluster.*)', Lines[LineIndex + 1])
				if (FF == None):
					LineIndex += 1
				else:
					break
			cd_hit_dict.update({dkey: li})
		LineIndex += 1

	return cd_hit_dict

def get_nr_proteins_list(protein_prefix : str, cd_hit_dict : dict):
	li = []
	for k in cd_hit_dict.keys():
		if(k.startswith(protein_prefix)):
			if(not(cd_hit_dict[k])):
				li.append(k)
			elif(all(protein.startswith(protein_prefix) for protein in cd_hit_dict[k])):
				li.append(k)
				li.extend(cd_hit_dict[k])
	return li

def generate_nr_test_set(test_set_orig_id_loc : str, test_set_orig_seq_loc : str, nr_protein_list : list):
	f = open (test_set_orig_id_loc, "r")
	p_data = json.load(f)
	f.close()

	pA = []
	pB = []
	pgr = nx.Graph()
	for i in range(len(p_data)):
		pA.append(p_data[i]['pA'])
		pB.append(p_data[i]['pB'])
		pgr.add_edge(p_data[i]['pA'], p_data[i]['pB'])

	to_remove = list(set(list(pgr.nodes())) - set(nr_protein_list))

	for i in range(len(to_remove)):
		pgr.remove_node(to_remove[i])

	f = open (test_set_orig_seq_loc, "r")
	seq_dict = json.load(f)
	f.close()

	seq_selected = {x : seq_dict[x] for x in nr_protein_list}

	return nx.to_pandas_edgelist(pgr), seq_selected
	
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

def generate_metrices_from_dict_single_type(orig_labels, predictions, pos_test_dict, save_preds_loc : str, interaction_type : str):
	print(f"Test accuracy: {np.mean(orig_labels == predictions)*100:.3f}")

	f = open(pos_test_dict)
	p_data = json.load(f)
	f.close()
	p_data_keys = list(p_data)

	fin_op = []
	fin_ip = []
	fin_op_thresh = []
	sub_img_count = 0
	for i in range(len(p_data)):
		img = p_data_keys[i]
		sub_img_count += len(p_data[img])
		if(sub_img_count > predictions.shape[0]):
			sub_img_count -= len(p_data[img])
			break
		elif(p_data[img]):
			outputs = [predictions[i - 1] for i in p_data[img]]
			fin_op.append(1 if np.mean(outputs)>=0.5 else 0)
			fin_ip.append(1)
			fin_op_thresh.append(np.mean(outputs))

	orig_labels = orig_labels[:sub_img_count]
	predictions = predictions[:sub_img_count]

	print("Final Image level prediction length : ", len(fin_op), len(fin_ip))

	correct = (np.array(fin_op) == np.array(fin_ip))
	accuracy = correct.sum() / correct.size
	print("Manual calculated Accuracy : ", accuracy*100)

	print("===============FROM SKLEARN=================")

	print("Accuracy = {}".format(accuracy))
	print("AUPRC = {}".format(average_precision_score(fin_ip, fin_op_thresh)))

	predSave_loc = save_preds_loc + "input_output_means.pkl"

	with open(predSave_loc, 'wb') as file:
		pickle.dump([fin_ip, fin_op, fin_op_thresh], file)

	print("All predictions saved at : ", predSave_loc)


def download_and_install_cd_hit(install_path : str):
	global cd_hit_command
	installed_object_path = install_path + 'cd-hit-v4.8.1-2019-0228/'
	if (os.path.exists(installed_object_path)):
		cd_hit_command =  installed_object_path + './cd-hit '
		print("cd-hit already installed in given location, setting global variable...")
		return
	out_loc = install_path + 'cd-hit-v4.8.1-2019-0228.tar.gz'
	print("Downloading cd-hit-v4.8.1-2019-0228")
	urllib.request.urlretrieve('https://github.com/weizhongli/cdhit/releases/download/V4.8.1/cd-hit-v4.8.1-2019-0228.tar.gz', out_loc)
	command = 'tar -xvzf ' + out_loc  + ' -C ' + install_path
	os.system(command)
	command = 'make -C ' + installed_object_path
	os.system(command)
	cd_hit_command =  installed_object_path + './cd-hit '

def choose_word_length_cd_hit(threshold : float):
	word_length = 0
	if (threshold <= 1 and threshold > 0.7) :
		word_length = 5
	elif (threshold <= 0.7 and threshold > 0.6) :
		word_length = 4
	elif (threshold <= 0.6 and threshold > 0.5) :
		word_length = 3
	elif (threshold <= 0.5) :
		word_length = 2
	return word_length

def remove_redundency_cd_hit(fasta_file_path : str, threshold : float, prefix : str):
	cpu_count = os.cpu_count()
	word_length = choose_word_length_cd_hit(threshold)
	out_path = os.path.abspath(os.path.join(fasta_file_path, '..')) + '/cd_hit_results/'
	createLocationIfNotExists(out_path)
	out_prefix = 'res_cd_hit_' + prefix
	cd_hit_config = ' -c ' + str(threshold) + ' -n ' + str(word_length) + '  -G 1 -g 1 -b 20 -l 10 -s 0.0 -aL 0.0 -aS 0.0 -T ' + str(cpu_count) + ' -M 32000'
	command = cd_hit_command + ' -i ' + fasta_file_path + ' -d 0 -o ' + out_path + out_prefix + cd_hit_config
	os.system(command)
	return out_prefix + '.clstr'

if __name__ == '__main__' :
	instal_path = '/home/aanzil/Documents/'
	download_and_install_cd_hit(instal_path)
	print("Testing cd-hit global command : ")
	os.system(cd_hit_command)
	print("==============================================")
	print("\nTesting done ......")
	fasta_file_loc = '/home/aanzil/Downloads/bench_celegans.fasta'
	print(remove_redundency_cd_hit(fasta_file_loc, 0.4))

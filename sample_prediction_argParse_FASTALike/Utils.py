import os, re, json, pickle, random, multiprocessing, scipy, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import average_precision_score
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm

SIZE = 128
STRIDE = 64
bs = 32
possible_sequences = ["M","D","A","K","R","G","L","C","V","F","S","P","Q","E","I","H","Y","T","W","N", "O", "U", "B", "Z", "X", "J"]
mat = {}
discard = ()

def set_global_stride(inStride : int):
	global STRIDE
	STRIDE = inStride

def trim_to_first_integer(s : str) -> str :
    match = re.search(r'\d', s)
    if match:
        return s[:match.start()]
    else:
        return s
	
def id_to_seq_dict_from_fasta(loc : str):
	fasta_sequences = SeqIO.parse(open(loc),'fasta')
	fasta_id_to_seq_dict = {}
	for fasta in fasta_sequences:
		name, sequence = fasta.id, str(fasta.seq)
		fasta_id_to_seq_dict[name] = sequence

	return fasta_id_to_seq_dict

def createLocationIfNotExists(loc : str):
	os.makedirs(loc, exist_ok=True)
	
class SubImageCounter:
	image_dict = {}
	subimage_counter = 0
	image_counter = 0

	def __init__(self) -> None:
		self.image_dict = {}
		self.subimage_counter = 0
		self.image_counter = 0

def checkChars(seq1, seq2):
    for v in discard:
        if (v in seq1):
            return False
        elif (v in seq2):
            return False
    return True

# PAM120 scoring matrix
def scoringMatrixPAM120():
	sMatrixTxt = '''
   A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y
A  3 -3  0  0 -4  1 -3 -1 -2 -3 -2  0  1 -1 -3  1  1  0 -7 -4
C -3  9 -7 -7 -6 -5 -4 -3 -7 -7 -6 -5 -3 -7 -4 -1 -3 -2 -8 -1
D  0 -7  5  3 -7  0  0 -3 -1 -5 -4  2 -2  1 -3  0 -1 -3 -8 -5
E  0 -7  3  5 -6 -1 -1 -3 -1 -4 -4  1 -1  2 -3 -1 -2 -3 -8 -4
F -4 -6 -7 -6  8 -5 -2  0 -6  0 -1 -4 -5 -6 -4 -3 -4 -3 -1  4
G  1 -5  0 -1 -5  5 -4 -4 -3 -5 -4  0 -2 -3 -4  1 -1 -2 -8 -6
H -3 -4  0 -1 -2 -4  7 -4 -2 -3 -4  2 -1  3  1 -2 -3 -3 -5 -1
I -1 -3 -3 -3  0 -4 -4  6 -2  1  1 -2 -3 -3 -2 -1  0  3 -7 -2
K -2 -7 -1 -1 -6 -3 -2 -2  5 -4  0  1 -2  0  2 -1 -1 -4 -5 -6
L -3 -7 -5 -4  0 -5 -3  1 -4  5  3 -4 -3 -2 -4 -4 -3  1 -5 -3
M -2 -6 -4 -4 -1 -4 -4  1  0  3  8 -3 -3 -1 -1 -2 -1  1 -7 -4
N  0 -5  2  1 -4  0  2 -2  1 -4 -3  4 -2  0 -1  1  0 -3 -5 -2
P  1 -3 -2 -1 -5 -2 -1 -3 -2 -3 -3 -2  6  0 -1  1 -1 -2 -7 -6
Q -1 -7  1  2 -6 -3  3 -3  0 -2 -1  0  0  6  1 -2 -2 -3 -6 -5
R -3 -4 -3 -3 -4 -4  1 -2  2 -4 -1 -1 -1  1  6 -1 -2 -2  1 -6
S  1 -1  0 -1 -3  1 -2 -1 -1 -4 -2  1  1 -2 -1  3 -6 -2 -2 -3
T  1 -3 -1 -2 -4 -1 -3  0 -1 -3 -1  0 -1 -2 -2 -6  4  0 -6 -3
V  0 -2 -3 -3 -3 -2 -3  3 -4  1  1 -3 -2 -3 -2 -2  0  5 -8 -3
W -7 -8 -8 -8 -1 -8 -5 -7 -5 -5 -7 -5 -7 -6  1 -2 -6 -8 12 -1
Y -4 -1 -5 -4  4 -6 -1 -2 -6 -3 -4 -2 -6 -5 -6 -3 -3 -3 -1  8
'''
	sMatrixList = sMatrixTxt.strip().split('\n')
	aaList = sMatrixList[0].split()
	sMatrix = dict()
	for aa in aaList:
		sMatrix[aa] = dict()
	for i in range(1, len(aaList) + 1):
		currRow = sMatrixList[i].split()
		for j in range(len(aaList)):
			sMatrix[currRow[0]][aaList[j]] = int(currRow[j + 1])
	return sMatrix

def substituton_matrix_init(out_dir : str, mat_type : str, mat_num : int):
	global mat, discard
	if (mat_type == "PAM" or mat_type == "pam"):
		if (mat_num == 120):
			mat = scoringMatrixPAM120()

	print("Choice for this session :")
	print("1 channel image saved using plt.imsave()...")
	mat_keys = list(dict(mat))

	residues = [e for e in set(mat_keys) if e.isalnum()]
	LUT = np.zeros((len(residues), len(residues)))
	for i in range(len(residues)):
		for j in range(len(residues)):
			LUT[i, j] = mat[residues[i]][residues[j]]
			
	print("Look Up Table Shape : ", LUT.shape)
	plt.imsave(out_dir + '_' + mat_type + '_' + str(mat_num) + 'LookUpTable.pdf', LUT)
	discard = set(possible_sequences) - set(residues)
	
def save_one_image_1C(it, fnameA, fnameB, seq1, seq2, base_loc):
	fin_img = np.zeros((len(seq1), len(seq2)))
		
	for i in range(len(seq1)):
		for j in range(len(seq2)):
			fin_img[i, j] = mat[seq1[i]][seq2[j]]
	
	file_name = str(base_loc + str(it+1) + "_" + fnameA + "_" + fnameB + ".png")
	plt.imsave(file_name, fin_img)
	
def sliding_window(image, stride, imgSize):
	height, width, _ = image.shape
	img = []
	a1 = list(range(0, height-imgSize+stride, stride))
	a2 = list(range(0, width-imgSize+stride, stride))
	if (a1[-1]+imgSize != height):
		a1[-1] = height-imgSize
	if (a2[-1]+imgSize != width):
		a2[-1] = width-imgSize
	for y in a1:
		for x in a2:
			im1 = image[y:y+imgSize, x:x+imgSize, :]
			img.append(np.array(im1))
	return img

def handle_one_image(path : str):
	image_data = []
	im = Image.open(path)
	im = np.array(im)
	img = sliding_window(im, STRIDE, SIZE)
	for i in range(len(img)):
		if(img[i].shape[2] >=3):
			image_data.append(img[i])
	return image_data

transformations = transforms.Compose([transforms.ToTensor()])
criterion = nn.CrossEntropyLoss()
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class PPIModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = torchvision.models.densenet201()
		self.model.classifier = nn.Linear(in_features = self.model.classifier.in_features, out_features = 2)

	def forward(self, x):
		x = self.model.forward(x)
		return x.squeeze()
	
def generate_metrices_from_dict_single_type(orig_labels, predictions, raw_pred, pos_test_dict, save_preds_loc : str):
	# Considers only positive
	print(f"Test accuracy: {np.mean(orig_labels == predictions)*100:.3f}")

	f = open(pos_test_dict)
	p_data = json.load(f)
	f.close()
	p_data_keys = list(p_data)

	tempImageCounter = SubImageCounter()
	tempImageCounter.image_counter += 1

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
			temp_raw_pred = [raw_pred[i - 1] for i in p_data[img]]
			tempImageCounter.image_dict[tempImageCounter.image_counter] = np.array(temp_raw_pred).tolist()
			tempImageCounter.image_counter += 1
			fin_op.append(1 if np.mean(outputs)>=0.5 else 0)
			fin_ip.append(1)
			fin_op_thresh.append(np.mean(outputs))

	dict_out_loc = save_preds_loc + "prediction_result_dict.json"

	with open(dict_out_loc,"w") as outfile:
		json.dump(tempImageCounter.image_dict, outfile)

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
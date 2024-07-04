import os, math, json, pickle, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import multiprocessing
from multiprocessing import Pool, cpu_count
import blosum as bl

maxAllowedSequenceLength = 300
SIZE = 128
STRIDE = 64
possible_sequences = ["M","D","A","K","R","G","L","C","V","F","S","P","Q","E","I","H","Y","T","W","N", "O", "U", "B", "Z", "X", "J"]
LUT_mat_3C =  np.zeros((len(possible_sequences), len(possible_sequences), 3))
map_aa2idxLUT = dict(zip(possible_sequences, range(0, len(possible_sequences))))
mat = {}
discard = ()
ch1flag = False

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

# PAM250 scoring matrix
def scoringMatrixPAM250():
	sMatrixTxt = '''
   A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y
A  2 -2  0  0 -3  1 -1 -1 -1 -2 -1  0  1  0 -2  1  1  0 -6 -3
C -2 12 -5 -5 -4 -3 -3 -2 -5 -6 -5 -4 -3 -5 -4  0 -2 -2 -8  0
D  0 -5  4  3 -6  1  1 -2  0 -4 -3  2 -1  2 -1  0  0 -2 -7 -4
E  0 -5  3  4 -5  0  1 -2  0 -3 -2  1 -1  2 -1  0  0 -2 -7 -4
F -3 -4 -6 -5  9 -5 -2  1 -5  2  0 -3 -5 -5 -4 -3 -3 -1  0  7
G  1 -3  1  0 -5  5 -2 -3 -2 -4 -3  0  0 -1 -3  1  0 -1 -7 -5
H -1 -3  1  1 -2 -2  6 -2  0 -2 -2  2  0  3  2 -1 -1 -2 -3  0
I -1 -2 -2 -2  1 -3 -2  5 -2  2  2 -2 -2 -2 -2 -1  0  4 -5 -1
K -1 -5  0  0 -5 -2  0 -2  5 -3  0  1 -1  1  3  0  0 -2 -3 -4
L -2 -6 -4 -3  2 -4 -2  2 -3  6  4 -3 -3 -2 -3 -3 -2  2 -2 -1
M -1 -5 -3 -2  0 -3 -2  2  0  4  6 -2 -2 -1  0 -2 -1  2 -4 -2
N  0 -4  2  1 -3  0  2 -2  1 -3 -2  2  0  1  0  1  0 -2 -4 -2
P  1 -3 -1 -1 -5  0  0 -2 -1 -3 -2  0  6  0  0  1  0 -1 -6 -5
Q  0 -5  2  2 -5 -1  3 -2  1 -2 -1  1  0  4  1 -1 -1 -2 -5 -4
R -2 -4 -1 -1 -4 -3  2 -2  3 -3  0  0  0  1  6  0 -1 -2  2 -4
S  1  0  0  0 -3  1 -1 -1  0 -3 -2  1  1 -1  0  2  1 -1 -2 -3
T  1 -2  0  0 -3  0 -1  0  0 -2 -1  0  0 -1 -1  1  3  0 -5 -3
V  0 -2 -2 -2 -1 -1 -2  4 -2  2  2 -2 -1 -2 -2 -1  0  4 -6 -2
W -6 -8 -7 -7  0 -7 -3 -5 -3 -2 -4 -4 -6 -5  2 -2 -5 -6 17  0
Y -3  0 -4 -4  7 -5  0 -1 -4 -1 -2 -2 -5 -4 -4 -3 -3 -2  0 10
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

def p_green_armytage_init(out_dir : str):	
	global LUT_mat_3C

	print("Choice for this session :")
	print("3 channel image saved using plt.imsave()...")

	colours = ["#FFCC99","#F0A3FF","#990000","#808080","#003380","#191919","#00998F",
		"#740AFF","#005C31","#94FFB5","#FFA405","#8F7C00","#FFFF80","#993F00",
		"#FFFF00","#C20088","#4C005C","#2BCE48","#FFFFFF","#FF5005","#0075DC",
		"#9DCC00","#FFA8BB","#426600","#FF0010","#5EF1F2"]
	for i in range(len(possible_sequences)):
		for j in range(len(possible_sequences)):
			cola = colours[map_aa2idxLUT[possible_sequences[i]]].lstrip('#')
			colb = colours[map_aa2idxLUT[possible_sequences[j]]].lstrip('#')
			r1, g1, b1 = tuple(int(cola[i:i+2], 16) for i in (0, 2, 4))
			r2, g2, b2 = tuple(int(colb[i:i+2], 16) for i in (0, 2, 4))
			LUT_mat_3C[i, j, 0] = math.sqrt((r1**2 + r2**2)/2)/255
			LUT_mat_3C[i, j, 1] = math.sqrt((g1**2 + g2**2)/2)/255
			LUT_mat_3C[i, j, 2] = math.sqrt((b1**2 + b2**2)/2)/255

	print("Look Up Table Shape : ", LUT_mat_3C.shape)
	plt.imsave(out_dir + 'p_green_armytage_LookUpTable.pdf', LUT_mat_3C)
	

def substituton_matrix_init(out_dir : str, mat_type : str, mat_num : int):
	global mat, discard, ch1flag
	if (mat_type == "PAM" or mat_type == "pam"):
		if (mat_num == 120):
			mat = scoringMatrixPAM120()
		elif (mat_num == 250):
			mat = scoringMatrixPAM250()
		else:
			print("Wrong PAM number!! please input from 120 or 250")
			exit(0)
	elif (mat_type == "BL" or mat_type == "bl"):
		if (mat_num == 45 or mat_num == 62 or mat_num == 90):
			mat = bl.BLOSUM(mat_num) 
		else:
			print("Wrong Blosum number!! please input from 45, 62 and 90")
			exit(0)
	else:
		print("Please choose from PAM, pam, BL, bl for using PAM and blosum 1D initialization")
		print("For PAM : mat_number, please input from 120 or 250")
		print("For Blosum : mat_number, please input from 45, 62 and 90")
		exit(0)

	ch1flag = True
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

def save_one_image_3C(it, fnameA, fnameB, seq1, seq2, base_loc):
	fin_img = np.zeros((len(seq1), len(seq2), 3))
		
	for i in range(len(seq1)):
		for j in range(len(seq2)):
			fin_img[i, j, ] = LUT_mat_3C[map_aa2idxLUT[seq1[i]], map_aa2idxLUT[seq2[j]],]
	
	file_name = str(base_loc + str(it+1) + "_" + fnameA + "_" + fnameB + ".png")
	plt.imsave(file_name, fin_img)

def generate_images(loc : str, interaction_type : str, out_loc : str):
	
	df = pd.read_pickle(loc)
	print(f"Generating {df.shape[0]} {interaction_type}, {1 if ch1flag else 3} channel images ...")
	base_loc = out_loc + interaction_type + "/"
	Path(base_loc).mkdir(parents=True, exist_ok=True)

	args = []
	for it in range(len(df)):
		args.append((it, df.iloc[it, 1], df.iloc[it, 2], df.iloc[it, 3], df.iloc[it, 4], base_loc))

	pool = Pool(cpu_count())
	if (ch1flag):
		pool.starmap(save_one_image_1C, args) 				#Synchronus but in parallel
	else:
		pool.starmap(save_one_image_3C, args)	

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
	
def gen_sub_img(in_dir : str, imgl : list, out_dir : str, interaction_type : str, imgCounter : SubImageCounter):

	args = []
	for img in imgl:
		args.append(in_dir + img)
			
	num_workers = multiprocessing.cpu_count()  

	with multiprocessing.Pool(processes = num_workers) as pool:
		for images in pool.map(handle_one_image, args):				#Synchronus but in parallel
			image_list = []
			for image in images:
				image_list.append(imgCounter.subimage_counter)
				file_name = str(out_dir + interaction_type + "_sub_" + str(imgCounter.subimage_counter) + ".png")
				plt.imsave(file_name, image)
				imgCounter.subimage_counter += 1
			imgCounter.image_dict[imgCounter.image_counter] = image_list
			imgCounter.image_counter += 1

	return imgCounter

def generate_sub_images(in_dir : str, imgl : list, out_dir : str, interaction_type : str, factor : int, dtype : str):
	
	print("Creating SubImages for ", interaction_type, ", ", dtype, " images...")

	base_loc = in_dir + interaction_type + "/"
	imagepath = ""
	if (interaction_type == "positive"):
		imagepath = out_dir + dtype + "/complete/pos_sub/"
	elif (interaction_type == "negative"):
		imagepath = out_dir + dtype + "/complete/neg_sub/"
	os.makedirs(imagepath, exist_ok=True)

	imgCounter = SubImageCounter()

	divImg = int(len(imgl)/factor)
	fromidx = 0
	toidx = 0
	for i in range(factor):
		if (i == (factor - 1)):
			fromidx, toidx = i*divImg, len(imgl)
		else:
			fromidx, toidx = i*divImg, (i+1)*divImg
		imgList = imgl[fromidx : toidx]
		imgCounter = gen_sub_img(base_loc, imgList, imagepath, interaction_type, imgCounter)
		print("Sliding Window, SubImage generation process finished for " + interaction_type + ", " + dtype + " data | part -> {}".format(i+1))

	image_dict_out_loc = out_dir + interaction_type + "_" + dtype + "_image_dict.json"

	with open(image_dict_out_loc,"w") as outfile:
		json.dump(imgCounter.image_dict, outfile)
	
	return imagepath, image_dict_out_loc

def generate_equal_len_sub_img_metadata(pos_sub_img_dict : str, neg_sub_img_dict : str, min_data_frac : float, out_dir : str, searchScopeChoice : str = "STD"):
	f = open(pos_sub_img_dict)
	pimage_dict = json.load(f)
	f.close()

	f = open(neg_sub_img_dict)
	nimage_dict = json.load(f)
	f.close()

	print("Positive dictionary size : ", len(pimage_dict.keys()))
	print("Negative dictionary size : ", len(nimage_dict.keys()))
	pos_sub_images = [len(pimage_dict[k]) for k in pimage_dict.keys()]
	neg_sub_images = [len(nimage_dict[k]) for k in nimage_dict.keys()]
	print("Positive Sub-image Count from dictionary :", sum(pos_sub_images))
	print("Positive Sub-image Count from dictionary :", sum(neg_sub_images))
	if searchScopeChoice == 'std' or searchScopeChoice == 'STD':
		searchScope = int(max(np.std(pos_sub_images), np.std(neg_sub_images)))
	elif searchScopeChoice == 'max' or searchScopeChoice == 'MAX':
		searchScope = int(min(np.max(pos_sub_images), np.max(neg_sub_images)))
	elif searchScopeChoice == 'median' or searchScopeChoice == 'MEDIAN':
		searchScope = int(min(np.median(pos_sub_images), np.median(neg_sub_images)))
	elif searchScopeChoice == 'mean' or searchScopeChoice == 'MEAN':
		searchScope = int(min(np.mean(pos_sub_images), np.mean(neg_sub_images)))
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

	print("Total images selected in positive train after Equalization : ", len(image_keys_p))
	print("Total images selected in negative train after Equalization : ", len(image_keys_n))
	print("Total sub-images in positive train after Equalization : ", len(sub_p))
	print("Total sub-images in negative train after Equalization : ", len(sub_n))
	print("Equalized with threshold : ", final_threshold)

	out_loc = out_dir + 'equal_train_data.pkl'

	with open(out_loc, 'wb') as file:
		pickle.dump([image_keys_p, image_keys_n, sub_p, sub_n], file)

	return out_loc

def copy_one_image(src : str, dest : str):
	if os.path.isfile(src):
		shutil.copy(src, dest)

def copy_equal_len_sub_images(equal_train_data_pkl_loc : str, pos_in_dir : str, neg_in_dir : str, out_dir : str):
	
	pool = Pool(cpu_count()) 

	with open(equal_train_data_pkl_loc, 'rb') as file:
		myvar = pickle.load(file)

	image_keys_p, image_keys_n, sub_p, sub_n = myvar

	pimagepath = out_dir + "/train/equal/pos_sub/"
	os.makedirs(pimagepath, exist_ok=True)
	nimagepath = out_dir+ "/train/equal/neg_sub/"
	os.makedirs(nimagepath, exist_ok=True)

	P_sub_img_list = ["positive_sub_"+ str(x) + ".png" for x in sub_p]
	N_sub_img_list = ["negative_sub_"+ str(x) + ".png" for x in sub_n]

	args = []
	for img in P_sub_img_list:
		full_file_name = os.path.join(pos_in_dir, img)
		args.append((full_file_name, pimagepath))

	pool.starmap(copy_one_image, args)

	print("Total Positive subimages copied : ", len(os.listdir(pimagepath)))

	args = []
	for img in N_sub_img_list:
		full_file_name = os.path.join(neg_in_dir, img)
		args.append((full_file_name, nimagepath))

	pool.starmap(copy_one_image, args)

	print("Total Negative subimages copied : ", len(os.listdir(nimagepath)))

	return pimagepath, nimagepath

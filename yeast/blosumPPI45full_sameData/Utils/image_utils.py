import os, math, json, pickle, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import multiprocessing
from multiprocessing import Pool, cpu_count
import blosum as bl

maxAllowedSequenceLength = 500
SIZE = 128
STRIDE = 64
possible_sequences = ["M","D","A","K","R","G","L","C","V","F","S","P","Q","E","I","H","Y","T","W","N", "O", "U", "B", "Z", "X", "J"]
look_up_table =  np.zeros((len(possible_sequences), len(possible_sequences), 3))
map_LUT = dict(zip(possible_sequences, range(0, len(possible_sequences))))
mat = {}
discard = ()

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

def blossom_init(blosum_num : int, out_dir : str):
	global mat, discard
	mat = bl.BLOSUM(blosum_num)    

	mat_keys = list(dict(mat))

	residues = [e for e in set(mat_keys) if e.isalnum()]
	LUT = np.zeros((len(residues), len(residues)))
	for i in range(len(residues)):
		for j in range(len(residues)):
			LUT[i, j] = mat[residues[i]][residues[j]]
			
	print("Look Up Table Shape : ", LUT.shape)
	plt.imsave(out_dir + 'blosum_' + str(blosum_num) + 'LookUpTable.pdf', LUT)
	discard = set(possible_sequences) - set(residues)

def save_one_image_blosum(it, fnameA, fnameB, seq1, seq2, base_loc):
	fin_img = np.zeros((len(seq1), len(seq2)))
		
	for i in range(len(seq1)):
		for j in range(len(seq2)):
			fin_img[i, j] = mat[seq1[i]][seq2[j]]
	
	file_name = str(base_loc + str(it+1) + "_" + fnameA + "_" + fnameB + ".png")
	plt.imsave(file_name, fin_img)

def generate_images_blosum(loc : str, interaction_type : str, out_loc : str):

	df = pd.read_pickle(loc)
	base_loc = out_loc + interaction_type + "/"
	Path(base_loc).mkdir(parents=True, exist_ok=True)

	print("Generating img type : ", interaction_type)

	args = []
	for it in range(len(df)):
		args.append((it, df.iloc[it, 1], df.iloc[it, 2], df.iloc[it, 3], df.iloc[it, 4], base_loc))

	pool = Pool(cpu_count()) 
	pool.starmap(save_one_image_blosum, args)
	

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
		for images in pool.map(handle_one_image, args):
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

def generate_equal_len_sub_img_metadata(pos_sub_img_dict : str, neg_sub_img_dict : str, min_data_frac : float, out_dir : str):
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
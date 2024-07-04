import os, pickle, random, re

split_num_finder = lambda x : int(re.search(r'[^\d]*(\d+)\.csv', x).group(1))

def createLocationIfNotExists(loc : str):
	os.makedirs(loc, exist_ok=True)

def createDataPartition(loc : str, data_split : float, out_dir : str):
	POSL=os.listdir("{}positive/".format(loc))
	NEGL=os.listdir("{}negative/".format(loc))
	
	print("Total: len(POS) = {}, len(NEG) = {}".format(len(POSL),len(NEGL)))

	P_sampled = random.sample(range(0, len(POSL)), len(POSL))
	N_sampled = random.sample(range(0, len(NEGL)), len(NEGL))

	HO_P_sampled = random.sample(range(0, len(POSL)), int(len(POSL) * (1 - data_split)))
	HO_N_sampled = random.sample(range(0, len(NEGL)), int(len(NEGL) * (1 - data_split)))

	P_sampled = [x for x in P_sampled if x not in HO_P_sampled]
	N_sampled = [x for x in N_sampled if x not in HO_N_sampled]

	HO_P_sampled = [POSL[x] for x in HO_P_sampled]
	HO_N_sampled = [NEGL[x] for x in HO_N_sampled]

	P_sampled = [POSL[x] for x in P_sampled]
	N_sampled = [NEGL[x] for x in N_sampled]

	print("Total: len(POS_train) = {}, len(NEG_train) = {}".format(len(P_sampled), len(N_sampled)))
	print("Total: len(POS_test) = {}, len(NEG_test) = {}".format(len(HO_P_sampled), len(HO_N_sampled)))
	
	out_loc = out_dir + 'data_train_test.pkl'

	with open(out_loc, 'wb') as file:
		pickle.dump([P_sampled, N_sampled, HO_P_sampled, HO_N_sampled], file)

	return out_loc

def createDataPartititionFromPrevDataCV(in_loc : str, in_data_cv_loc : str, out_dir : str):
	POSL=os.listdir("{}positive/".format(in_loc))
	NEGL=os.listdir("{}negative/".format(in_loc))
	
	print("Total: POS={},NEG={}".format(len(POSL),len(NEGL)))

	with open(in_data_cv_loc, 'rb') as file:
		myvar = pickle.load(file)

	HO_P_sampled = myvar[2]
	HO_N_sampled = myvar[3]

	res = []
	for ho in HO_P_sampled :
		str_to_end_with = ho.split('_', 1)[1]
		for img in POSL :
			if img.endswith(str_to_end_with):
				res.append(img)
				break
	HO_P_sampled = res.copy()

	res = []
	for ho in HO_N_sampled :
		str_to_end_with = ho.split('_', 1)[1]
		for img in NEGL :
			if img.endswith(str_to_end_with):
				res.append(img)
				break
	HO_N_sampled = res.copy()

	P_sampled = [x for x in POSL if x not in HO_P_sampled]
	N_sampled = [x for x in NEGL if x not in HO_N_sampled]

	print("Total: len(POS_train) = {}, len(NEG_train) = {}".format(len(P_sampled), len(N_sampled)))
	print("Total: len(POS_test) = {}, len(NEG_test) = {}".format(len(HO_P_sampled), len(HO_N_sampled)))
	
	out_loc = out_dir + 'data_train_test.pkl'

	with open(out_loc, 'wb') as file:
		pickle.dump([P_sampled, N_sampled, HO_P_sampled, HO_N_sampled], file)

	return out_loc
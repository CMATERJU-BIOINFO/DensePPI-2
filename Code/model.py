lr = 1e-4
bs = 32
num_epochs = 10
MOMENTUM = 0.9
epochs = 10

import pickle, json, random
import torch, os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from Utils.model_utils import *
from Utils.image_utils import SubImageCounter
from tqdm import tqdm

num_workers = int(os.cpu_count()/4)
transformations = transforms.Compose([transforms.ToTensor()])
criterion = nn.CrossEntropyLoss()
device = None
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def set_global_device(device_name : str):
	global device
	device = torch.device(device_name if torch.cuda.is_available() else "cpu") # 1 for sc, 0 for pea
	print("Starting session using device : ", device)

class PPIModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = torchvision.models.densenet201()
		self.model.classifier = nn.Linear(in_features = self.model.classifier.in_features, out_features = 2)

	def forward(self, x):
		x = self.model.forward(x)
		return x.squeeze()
	
def train(train_sub_images_parent_loc : str, model_save_path : str, model_type : str, intermediate = False, saved_model_file = ""):
	train_data = datasets.ImageFolder(train_sub_images_parent_loc, transform=transformations)
	train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

	modelcheckpoint = ModelCheckpoint(model_save_load_dir=model_save_path, model_save_prefix=model_type, mode='max')

	model = PPIModel()
	model = model.to(device=device)
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)

	if (intermediate == True):
		modelcheckpoint.load_checkpoint(saved_model_file, model, optimizer)
		
	
	train_loss = 0
	train_acc = 0

	for epoch in range(modelcheckpoint.get_num_epochs(), epochs):
		model.train()
		with tqdm(train_dataloader, unit="batch") as tepoch:
			tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
			for input, labels in tepoch:
				input, labels = input.to(device), labels.to(device)
				optimizer.zero_grad()
				logits = model.forward(input)
				loss = criterion(logits, labels)
				loss.backward()
				optimizer.step()
				train_loss += loss.item()
				ps = torch.exp(logits)
				_, top_class = ps.topk(1, dim=1)
				equals = top_class == labels.view(*top_class.shape)
				acc = torch.mean(equals.type(torch.FloatTensor)).item()  
				train_acc += acc
				tepoch.set_postfix(tr_loss = loss.item(), tr_acc = float(acc))

		print(
			f"Epoch {epoch + 1}/{epochs} : "
			f"Train loss : {train_loss/len(train_dataloader)} | "
			f"Train Accuracy : {train_acc/len(train_dataloader)}"
		)

		modelcheckpoint.create_checkpoint(train_acc/len(train_dataloader), model, optimizer)
		train_loss = 0
		train_acc = 0

	final_save_loc = modelcheckpoint.get_last_saved_loc()
	print("Training Finished...")
	print("Final trained model for " + model_type + " type model is saved at : " + final_save_loc)
	return final_save_loc

def predict(test_sub_images_parent_loc : str, model_save_path : str):
	test_data = datasets.ImageFolder(test_sub_images_parent_loc, transform=transformations)
	if (len(test_data.classes) == 1 and test_data.classes[0] == 'pos_sub'): #considering only positive
		test_data.samples = [(d, 1) for d, _ in test_data.samples]
	test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=bs, num_workers=num_workers, pin_memory=True, drop_last=True)

	modelcheckpoint = ModelCheckpoint(model_save_load_dir=os.path.dirname(model_save_path))
	model = PPIModel()
	model = model.to(device=device)
	modelcheckpoint.load_checkpoint(model_save_path.split('/')[-1], model)
	model.eval()

	gr_tr = []
	pred = []
	raw_pred = []
	test_loss = 0
	test_acc = 0

	with torch.no_grad(), tqdm(test_dataloader, unit="batch") as tepoch:
		tepoch.set_description(f"Predicting ")
		for input, labels in tepoch:
			input, labels = input.to(device), labels.to(device)
			logits = model.forward(input)
			loss = criterion(logits, labels)
			test_loss += loss.item()
			ps = torch.exp(logits)
			_, top_class = ps.topk(1, dim=1)
			equals = top_class == labels.view(*top_class.shape)
			acc = torch.mean(equals.type(torch.FloatTensor)).item()  
			gr_tr.extend(labels.tolist())
			raw_pred.extend(logits.tolist())
			pred.extend(top_class.squeeze().tolist())
			test_acc += acc
			tepoch.set_postfix(tst_loss = loss.item(), tst_acc = float(acc))

	print(
		f"Prediction loss : {test_loss/len(test_dataloader)} | "
		f"Prediction Accuracy : {test_acc/len(test_dataloader)}"
	)

	return np.array(gr_tr), np.array(pred), np.array(raw_pred)

def generate_metrices_from_dict(orig_labels, predictions, raw_pred, pos_test_dict, neg_test_dict, save_preds_loc : str):
	print(f"Test accuracy: {np.mean(orig_labels == predictions):.5f}")

	f = open(pos_test_dict)
	p_data = json.load(f)
	f.close()

	f = open(neg_test_dict)
	n_data = json.load(f)
	f.close()

	tempImageCounter = SubImageCounter()
	tempImageCounter.image_counter += 1

	idx = 1
	fin_op = []
	fin_ip = []
	fin_op_thresh = []
	sub_img_count = 0
	for img in n_data:
		sub_img_count += len(n_data[img])
		
		if(n_data[img]):
			outputs = [predictions[idx + i - 1] for i in n_data[img]]
			temp_raw_pred = [raw_pred[idx + i - 1] for i in n_data[img]]
			tempImageCounter.image_dict[tempImageCounter.image_counter] = np.array(temp_raw_pred).tolist()
			tempImageCounter.image_counter += 1
			fin_op.append(1 if np.mean(outputs)>=0.55 else 0)
			fin_ip.append(0)
			fin_op_thresh.append(np.mean(outputs))

	dict_out_loc = save_preds_loc + "negative_prediction_result_dict.json"

	with open(dict_out_loc,"w") as outfile:
		json.dump(tempImageCounter.image_dict, outfile)

	tempImageCounter = SubImageCounter()
	tempImageCounter.image_counter += 1

	idx = n_data[str(len(n_data)-1)][-1]
	for img in p_data:
		sub_img_count += len(p_data[img])
		if(sub_img_count > predictions.shape[0]):
			sub_img_count -= len(p_data[img])
			break
		elif(p_data[img]):
			outputs = [predictions[idx + i - 1] for i in p_data[img]]
			temp_raw_pred = [raw_pred[idx + i - 1] for i in p_data[img]]
			tempImageCounter.image_dict[tempImageCounter.image_counter] = np.array(temp_raw_pred).tolist()
			tempImageCounter.image_counter += 1
			fin_op.append(1 if np.mean(outputs)>=0.55 else 0)
			fin_ip.append(1)
			fin_op_thresh.append(np.mean(outputs))

	dict_out_loc = save_preds_loc + "positive_prediction_result_dict.json"

	with open(dict_out_loc,"w") as outfile:
		json.dump(tempImageCounter.image_dict, outfile)

	orig_labels = orig_labels[:sub_img_count]
	predictions = predictions[:sub_img_count]

	print("Final Image level prediction length : ", len(fin_op), len(fin_ip))
	print("Final sub-image level prediction length : ", sub_img_count)

	correct = (np.array(fin_op) == np.array(fin_ip))
	accuracy = correct.sum() / correct.size
	print("Manual calculated Accuracy : ", accuracy)

	CM = confusion_matrix(fin_ip, fin_op)
	TN = CM[0][0]
	FN = CM[1][0]
	TP = CM[1][1]
	FP = CM[0][1]
	
	print("TN = {}".format(TN))
	print("FP = {}".format(FP))
	print("TP = {}".format(TP))
	print("FN = {}".format(FN))

	print("===============FROM SKLEARN=================")

	print("Accuracy = {}".format(accuracy))
	print("AUPRC = {}".format(average_precision_score(fin_ip, fin_op_thresh)))
	print("ROC AUC = {}".format(roc_auc_score(fin_ip, fin_op_thresh)))
	print("MCC = {}".format(matthews_corrcoef(fin_ip, fin_op)))
	print("Sensitivity = {}".format(TP/(TP+FN)))
	print("Specificity = {}".format(TN/(TN+FP)))
	print("Precision = {}".format(TP/(TP+FP)))
	precision, recall, _ = precision_recall_curve(fin_ip, fin_op_thresh)
	f1, PRauc = f1_score(fin_ip, fin_op), auc(recall, precision)
	print("F1 Score = {}".format(f1))
	print("PR AUC = {}".format(PRauc))

	fpr, tpr, _ = roc_curve(fin_ip, fin_op_thresh)
	plt.plot(fpr, tpr, label='ROC AUC')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend()
	plt.savefig(save_preds_loc + 'roc_curve.pdf')
	plt.close()

	plt.plot(recall, precision, label='PR AUC')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend()
	plt.savefig(save_preds_loc + 'auc_curve.pdf')
	plt.close()


	predSave_loc = save_preds_loc + "input_output_means.pkl"

	with open(predSave_loc, 'wb') as file:
		pickle.dump([fin_ip, fin_op, fin_op_thresh], file)

	print("All predictions saved at : ", predSave_loc)

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
			fin_op.append(1 if np.mean(outputs)>=0.55 else 0)
			fin_ip.append(1)
			fin_op_thresh.append(np.mean(outputs))

	dict_out_loc = save_preds_loc + "positive_prediction_result_dict.json"

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
import sys, pickle

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from Utils.image_utils import *
from Utils.path_utils import *
from Utils.data_read_utils import *
from model import *

def main():
	set_global_device("cuda:0")
	project_base_dir = abspath(join(dirname(__file__), '..'))
	images_generated = True

	positive_image_txt_loc = "{}/input/PanEtAlData/Supp-A.txt".format(project_base_dir)
	negative_image_txt_loc = "{}/input/PanEtAlData/Supp-B.txt".format(project_base_dir)
	out_dir = "{}/outputs/SCerevisiae/".format(project_base_dir) #Use end slash

	
	#createLocationIfNotExists(out_dir)
			
	pos_train_sub_img_loc, pos_train_sub_img_dict, neg_train_sub_img_loc, neg_train_sub_img_dict, pos_test_sub_img_loc, pos_test_sub_img_dict, neg_test_sub_img_dict, equal_train_data_pkl_loc, eq_neg_train_sub_img_loc = process_and_return_dirs(out_dir)
	
	test_parent_loc = os.path.abspath(os.path.join(pos_test_sub_img_loc, '..'))
	
	modelSaveDir = "saved_models"
	modelSavePath = os.path.join(out_dir, modelSaveDir)

	print("Training and Testing Complete Model")

	# Full Model
	outputSavePath = out_dir + "finalPredsComplete_"
	train_parent_loc = os.path.abspath(os.path.join(pos_train_sub_img_loc, '..'))
	# model_saved_loc = train(train_parent_loc, modelSavePath, "complete")
	#model_saved_loc = train(train_parent_loc, modelSavePath, "complete", True, "complete_epoch_009_metric_0.98070.pth.tar")
	model_saved_loc = "/media/ppin-1/DISC-1/test_code_integrity_aanzil/blosumPPI/JUPPIdBlosum45PPIComp/outputs/blosumJUPPId/C1_10_Fold/fold1/saved_models/complete_epoch_010_metric_0.99531.pth.tar"

	orig_labels, predictions = predict(test_parent_loc, model_saved_loc)     
	generate_metrices_from_dict(orig_labels, predictions, pos_test_sub_img_dict, neg_test_sub_img_dict, outputSavePath)
	
	print("Training and Testing Complete Model Finished....")
	


if __name__ == '__main__':
	main()

import sys, pickle

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../..')))

from Utils.image_utils import *
from Utils.path_utils import *
from Utils.data_read_utils import *
from model import *

def main():
	set_global_device("cuda:0")
	project_base_dir = abspath(join(dirname(__file__), '../..'))
	curr_dir = abspath(dirname(__file__))
	images_generated = False
	
	out_dir = "{}/outputs/SCerevisiae/".format(curr_dir) #Use end slash
	standard_save_loc = "{}/input/sCerevisiaeData/standard/".format(project_base_dir)
	std_data_cv_loc = "{}data_train_test.pkl".format(standard_save_loc)
	orig_img_pos_out_pkl_loc = standard_save_loc + 'positive.pkl'
	orig_img_neg_out_pkl_loc = standard_save_loc + 'negative.pkl'

	if (not images_generated):
		createLocationIfNotExists(out_dir)
		substituton_matrix_init(out_dir, 'pam', 250)
		generate_images(orig_img_pos_out_pkl_loc, 'positive', out_dir)
		generate_images(orig_img_neg_out_pkl_loc, 'negative', out_dir)
		
		train_test_pkl_loc =  createDataPartititionFromPrevDataCV(out_dir, std_data_cv_loc, out_dir)

  
		with open(train_test_pkl_loc, 'rb') as file:
			myvar = pickle.load(file)

		pos_train_img_list = myvar[0]
		neg_train_img_list = myvar[1]
		pos_test_img_list = myvar[2]
		neg_test_img_list = myvar[3]

		pos_train_sub_img_loc, pos_train_sub_img_dict = generate_sub_images(out_dir, pos_train_img_list, out_dir, "positive", 5, "train")
		neg_train_sub_img_loc, neg_train_sub_img_dict = generate_sub_images(out_dir, neg_train_img_list, out_dir, "negative", 5, "train")
		pos_test_sub_img_loc, pos_test_sub_img_dict = generate_sub_images(out_dir, pos_test_img_list, out_dir, "positive", 2, "test")
		_, neg_test_sub_img_dict = generate_sub_images(out_dir, neg_test_img_list, out_dir, "negative", 2, "test")
		
		equal_train_data_pkl_loc = generate_equal_len_sub_img_metadata(pos_train_sub_img_dict, neg_train_sub_img_dict, 0.2, out_dir, 'max')
		_, eq_neg_train_sub_img_loc = copy_equal_len_sub_images(equal_train_data_pkl_loc, pos_train_sub_img_loc, neg_train_sub_img_loc, out_dir)
	else:
		pos_train_sub_img_loc, pos_train_sub_img_dict, neg_train_sub_img_loc, neg_train_sub_img_dict, pos_test_sub_img_loc, pos_test_sub_img_dict, neg_test_sub_img_dict, equal_train_data_pkl_loc, eq_neg_train_sub_img_loc = process_and_return_dirs(out_dir)

	test_parent_loc = os.path.abspath(os.path.join(pos_test_sub_img_loc, '..'))
	
	modelSaveDir = "saved_models"
	modelSavePath = os.path.join(out_dir, modelSaveDir)

	print("Training and Testing Complete Model")

	# Full Model
	outputSavePath = out_dir + "finalPredsComplete_"
	train_parent_loc = os.path.abspath(os.path.join(pos_train_sub_img_loc, '..'))
	model_saved_loc = train(train_parent_loc, modelSavePath, "complete")
	orig_labels, predictions, raw_preds = predict(test_parent_loc, model_saved_loc)     
	generate_metrices_from_dict(orig_labels, predictions, raw_preds, pos_test_sub_img_dict, neg_test_sub_img_dict, outputSavePath)
	generate_test_results_to_csv(std_data_cv_loc, outputSavePath, standard_save_loc, outputSavePath)	

	print("Training and Testing Complete Model Finished....")
	print("============================================================")
	print("Training and Testing Equal Model")
	
	# Equal Model
	outputSavePath = out_dir + "finalPredsEqual_"
	train_parent_loc = os.path.abspath(os.path.join(eq_neg_train_sub_img_loc, '..'))
	model_saved_loc = train(train_parent_loc, modelSavePath, "equal")
	orig_labels, predictions, raw_preds = predict(test_parent_loc, model_saved_loc)     
	generate_metrices_from_dict(orig_labels, predictions, raw_preds, pos_test_sub_img_dict, neg_test_sub_img_dict, outputSavePath)
	generate_test_results_to_csv(std_data_cv_loc, outputSavePath, standard_save_loc, outputSavePath)	
 
	print("Training and Testing Equal Model Finished....")


if __name__ == '__main__':
	main()
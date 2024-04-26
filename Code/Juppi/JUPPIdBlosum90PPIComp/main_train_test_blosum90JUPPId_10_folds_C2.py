# %%
import sys

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from Utils.image_utils import *
from Utils.path_utils import *
from Utils.data_read_utils import *
from model import *
from pathlib import Path

# %%
set_global_device("cuda:0")
project_base_dir = abspath(join(dirname(__file__), '..'))
images_generated = False
curr_dir = abspath(dirname(__file__))

testing_input_subfolder_name = "C2_10_Fold"
input_fasta_seq_file = "{}/input/JUPPId/human_JUPPI.seq".format(project_base_dir)
input_interaction_train_test_dir = "{}/input/JUPPId/{}/".format(project_base_dir, testing_input_subfolder_name)
out_dir = "{}/outputs/blosumJUPPId/{}/".format(curr_dir, testing_input_subfolder_name) #Use end slash

# %%
if (not images_generated):
        createLocationIfNotExists(out_dir)
        substituton_matrix_init(out_dir, 'bl', 90)

        train_files_iterator = Path(input_interaction_train_test_dir).glob("Train*")
        train_csvs_loc = [str(files) for files in train_files_iterator]
        train_csvs_num = [split_num_finder(f) for f in train_csvs_loc]
        train_csv_num_to_loc_dict = dict(zip(train_csvs_num, train_csvs_loc))

        test_files_iterator = Path(input_interaction_train_test_dir).glob("Test*")
        test_csvs_loc = [str(files) for files in test_files_iterator]
        test_csvs_num = [split_num_finder(f) for f in test_csvs_loc]
        test_csv_num_to_loc_dict = dict(zip(test_csvs_num, test_csvs_loc))

        total_splits = sorted(test_csv_num_to_loc_dict.keys())

        for split in total_splits:
            print("==============================================")
            print("Training and Testing started for fold : {}".format(split))
            print("==============================================")
            temp_out_dir = "{}/fold{}/".format(out_dir, split)
            createLocationIfNotExists(temp_out_dir)
            a, b, c, d, e, f = create_train_test_imgs_and_return_split(train_csv_num_to_loc_dict[split], test_csv_num_to_loc_dict[split], input_fasta_seq_file, temp_out_dir)
            img_dir_train, img_dir_test = a, b
            pos_train_img_list, neg_train_img_list = c, d
            pos_test_img_list, neg_test_img_list = e, f
            
            pos_train_sub_img_loc, pos_train_sub_img_dict = generate_sub_images(img_dir_train, pos_train_img_list, temp_out_dir, "positive", 5, "train")
            neg_train_sub_img_loc, neg_train_sub_img_dict = generate_sub_images(img_dir_train, neg_train_img_list, temp_out_dir, "negative", 5, "train")
            pos_test_sub_img_loc, pos_test_sub_img_dict = generate_sub_images(img_dir_test, pos_test_img_list, temp_out_dir, "positive", 2, "test")
            _, neg_test_sub_img_dict = generate_sub_images(img_dir_test, neg_test_img_list, temp_out_dir, "negative", 2, "test")        
            break
else:
		pos_train_sub_img_loc, pos_train_sub_img_dict, neg_train_sub_img_loc, neg_train_sub_img_dict, pos_test_sub_img_loc, pos_test_sub_img_dict, neg_test_sub_img_dict, equal_train_data_pkl_loc, eq_neg_train_sub_img_loc = process_and_return_dirs1(out_dir)	
    
test_parent_loc = os.path.abspath(os.path.join(pos_test_sub_img_loc, '..'))
modelSaveDir = "saved_models"
temp_out_dir = out_dir + 'fold1/'
modelSavePath = os.path.join(temp_out_dir, modelSaveDir)

print("Training and Testing Complete Model")
# pos_test_sub_img_dict = temp_out_dir + "positive" + "_" + "test" + "_image_dict.json"
# neg_test_sub_img_dict = temp_out_dir + "negative" + "_" + "test" + "_image_dict.json"
    # Full Model
outputSavePath = temp_out_dir + "finalPredsComplete_"
train_parent_loc = os.path.abspath(os.path.join(pos_train_sub_img_loc, '..'))
model_saved_loc = train(train_parent_loc, modelSavePath, "complete")
# model_saved_loc = train(train_parent_loc, modelSavePath, "complete", True, "complete_epoch_009_metric_0.99365.pth.tar")
orig_labels, predictions = predict(test_parent_loc, model_saved_loc)     
generate_metrices_from_dict(orig_labels, predictions, pos_test_sub_img_dict, neg_test_sub_img_dict, outputSavePath)

print("Training and Testing Complete Model Finished....")


import sys

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../..')))

from Utils.image_utils import *
from Utils.path_utils import *
from Utils.data_read_utils import *
from model import *

def main():
    set_global_device("cuda:1")
    project_base_dir = abspath(join(dirname(__file__), '../..'))
    images_generated = True
    curr_dir = abspath(dirname(__file__))

    testing_input_subfolder_name = "C3Set1"
    input_fasta_seq_file = "{}/input/JUPPId/human_JUPPI.seq".format(project_base_dir)
    input_interaction_train_test_dir = "{}/input/JUPPId/{}/".format(project_base_dir, testing_input_subfolder_name)
    out_dir = "{}/outputs/blosum90JUPPId/{}/".format(curr_dir, testing_input_subfolder_name) #Use end slash

    if (not images_generated):
        train_csv_loc = input_interaction_train_test_dir + "TrainSet.csv"
        test_csv_loc = input_interaction_train_test_dir + "TestSet.csv"
        createLocationIfNotExists(out_dir)
        substituton_matrix_init(out_dir, 'bl', 90)
        createLocationIfNotExists(out_dir)
        a, b, c, d, e, f = create_train_test_imgs_and_return_split(train_csv_loc, test_csv_loc, input_fasta_seq_file, out_dir)
        img_dir_train, img_dir_test = a, b
        pos_train_img_list, neg_train_img_list = c, d
        pos_test_img_list, neg_test_img_list = e, f
        
        pos_train_sub_img_loc, _ = generate_sub_images(img_dir_train, pos_train_img_list, out_dir, "positive", 5, "train")
        _, _ = generate_sub_images(img_dir_train, neg_train_img_list, out_dir, "negative", 5, "train")
        pos_test_sub_img_loc, pos_test_sub_img_dict = generate_sub_images(img_dir_test, pos_test_img_list, out_dir, "positive", 2, "test")
        _, neg_test_sub_img_dict = generate_sub_images(img_dir_test, neg_test_img_list, out_dir, "negative", 2, "test")        
    else:
        pos_train_sub_img_loc, _, _, _, pos_test_sub_img_loc, pos_test_sub_img_dict, neg_test_sub_img_dict, _, _ = process_and_return_dirs(out_dir, False)	
        
    test_parent_loc = os.path.abspath(os.path.join(pos_test_sub_img_loc, '..'))
    modelSaveDir = "saved_models"
    modelSavePath = os.path.join(out_dir, modelSaveDir)

    print("Training and Testing Complete Model")
    outputSavePath = out_dir + "finalPredsComplete_"
    train_parent_loc = os.path.abspath(os.path.join(pos_train_sub_img_loc, '..'))
    model_saved_loc = train(train_parent_loc, modelSavePath, "complete")
    orig_labels, predictions, raw_preds = predict(test_parent_loc, model_saved_loc)     
    generate_metrices_from_dict(orig_labels, predictions, raw_preds, pos_test_sub_img_dict, neg_test_sub_img_dict, outputSavePath)

    print("Training and Testing Complete Model Finished....")


if __name__ == '__main__':
	main()
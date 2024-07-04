import sys

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../..')))

from Utils.image_utils import *
from Utils.path_utils import *
from Utils.data_read_utils import *
from model import *

def main():
	set_global_device("cuda:1")
	test_model_type = "C2Set1"
	project_base_dir = abspath(join(dirname(__file__), '../..'))
	out_dir = "{}/outputs/SarsCov2/{}/".format(project_base_dir, test_model_type) #Use end slash
	input_data_base_loc = "{}/input/SarsCov2Data/".format(project_base_dir)
	final_image_df_loc = "{}final_test.pkl".format(out_dir)
	createLocationIfNotExists(out_dir)
	substituton_matrix_init(out_dir, 'pam', 120)
	this_dict_p1_p2 = input_data_base_loc + "positive_SarsCov2_IDs.json"
	this_dict_id_to_sec_loc = input_data_base_loc + "positive_SarsCov2_P1_P2.json"
	final_edges_test, final_seq = get_pandas_edge_list_from_json_paths(this_dict_p1_p2, this_dict_id_to_sec_loc)

	parse_images_from_edgeList_seq(final_edges_test, final_seq, final_image_df_loc)
	generate_images(final_image_df_loc, 'positive', out_dir)

	POSL=os.listdir("{}/positive/".format(out_dir))
	print("Total: len(POS) = {}".format(len(POSL)))

	pos_test_sub_img_loc, pos_test_sub_img_dict = generate_sub_images(out_dir, POSL, out_dir, "positive", 5, "test")
	test_parent_loc = os.path.abspath(os.path.join(pos_test_sub_img_loc, '..'))

	outputSavePath = out_dir + "finalPredsComplete_"
	model_saved_loc = "/media/ppin-1/DISC-1/test_code_integrity_aanzil/blosumPPI/JUPPIdBlosum45PPIComp/outputs/blosumJUPPId/C1_10_Fold/fold1/saved_models/complete_epoch_010_metric_0.99531.pth.tar"

	orig_labels, predictions, raw_preds = predict(test_parent_loc, model_saved_loc)     
	generate_metrices_from_dict_single_type(orig_labels, predictions, raw_preds, pos_test_sub_img_dict, outputSavePath)


if __name__ == '__main__':
	main()
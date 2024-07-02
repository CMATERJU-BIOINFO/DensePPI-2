import pickle
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

encoding_strategy = "densePPI"
key_loc  = {
    "Dense" : "________/JUPPIdDensePPIComp/outputs/blosumJUPPId/C1_10_Fold/fold1/finalPredsComplete_input_output_means.pkl",
    "Blosum45" : "________/JUPPIdBlosum45PPIComp/outputs/blosumJUPPId/C1_10_Fold/fold1/finalPredsComplete_input_output_means.pkl",
    "Blosum90" : "________/JUPPIdBlosum90PPIComp/outputs/blosumJUPPId/C1_10_Fold/fold1/finalPredsComplete_input_output_means.pkl",
    "PAM120" : "________/JUPPIdPAM120PPIComp/outputs/blosumJUPPId/C1_10_Fold/fold1/finalPredsComplete_input_output_means.pkl",
    "PAM250" : "________/JUPPIdPAM250PPIComp/outputs/blosumJUPPId/C1_10_Fold/fold1/finalPredsComplete_input_output_means.pkl"
}

plt.figure()
plt.title(' ')
for data_type, loc in key_loc.items():
    fin_ip, fin_op, fin_op_thresh = None, None, None
    with open(loc, 'rb') as f:
        fin_ip, fin_op, fin_op_thresh = pickle.load(f)
    fpr, tpr, threshold = metrics.roc_curve(fin_ip, fin_op_thresh)
    plt.plot(fpr, tpr, label = data_type + '_AUC = %0.5f' % metrics.auc(fpr, tpr))
plt.plot([0, 1], [0, 1], '--', label = "Random")
plt.legend(loc = 'lower right')
plt.xlim([-0.02, 1])
plt.ylim([0, 1.02])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('ROC Curve for Juppi C1 Dataset', dpi = 600)
plt.show()

plt.figure()
plt.title(' ')
default_bar = 0
for data_type, loc in key_loc.items():
    fin_ip, fin_op, fin_op_thresh = None, None, None
    with open(loc, 'rb') as f:
        fin_ip, fin_op, fin_op_thresh = pickle.load(f)
    if (default_bar ==  0):
        temp = np.array(fin_ip)
        default_bar = len(temp[temp == 1]) / len(temp)
    precision, recall, thresholds = metrics.precision_recall_curve(fin_ip, fin_op_thresh)
    plt.plot(recall, precision, label = data_type + '_AUC = %0.5f' % metrics.auc(recall, precision))
plt.plot([0, 1], [default_bar, default_bar], '--', label = "Random")
plt.legend(loc = 'lower left')
plt.xlim([0, 1.02])
plt.ylim([default_bar - 0.02, 1.02])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.savefig('PRC Curve for Juppi C1 Dataset', dpi = 600)
plt.show()
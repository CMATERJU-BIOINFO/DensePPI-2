import numpy as np, os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from PIL import Image

# ### Normalization Neighbouring Layer with Deay :

slwCount = lambda height, width, window,stride: ((height - window) // stride + 1 )*((width - window) // stride + 1 )
extract_file_info = lambda s: (s.split('_')[0].upper(), ''.join([char for char in s if char.isdigit()])[-2:])


def normalize(predictions):
    flat_predictions = np.array(predictions).flatten()
    min_val = np.min(flat_predictions)
    max_val = np.max(flat_predictions)
    normalized_predictions = 2 * (np.array(predictions) - min_val) / (max_val - min_val) - 1
    return normalized_predictions.tolist()

def decay_function(value, layer, decay_rate=0.1):
    return value * np.exp(-decay_rate * layer)

def create_centered_matrix(m, v, decay_rate=0.1):
    matrix = np.zeros((m, m), dtype=float)
    center = m // 2

    for layer in range(center + 1):
        value = decay_function(v, layer, decay_rate)

        if center - layer >= 0 and center + layer < m:
            matrix[center - layer, center - layer:center + layer + 1] = value[center - layer, center - layer:center + layer + 1]
            matrix[center + layer, center - layer:center + layer + 1] = value[center + layer, center - layer:center + layer + 1]
            matrix[center - layer:center + layer + 1, center - layer] = value[center - layer:center + layer + 1, center - layer]
            matrix[center - layer:center + layer + 1, center + layer] = value[center - layer:center + layer + 1, center + layer]

    return matrix

# ### Image Gen

def InteractionMapPlotComplete(pid,xp,yp,seq1,seq2, predictions, pred_img_dims, height, width, imgSize, stride,opf, norm=False, decay_rate=0.1):

    a1 = list(range(0, height-imgSize+stride, stride))
    a2 = list(range(0, width-imgSize+stride, stride))
    if (a1[-1]+imgSize != height):
        a1[-1] = height-imgSize
    if (a2[-1]+imgSize != width):
        a2[-1] = width-imgSize

    fin_img = np.zeros((height, width))

    assert pred_img_dims[0] == len(a1)
    assert pred_img_dims[1] == len(a2)

    flat_predictions = np.array(predictions).flatten()
    vmax = np.max(flat_predictions)
    vmin = np.min(flat_predictions)
    predIdxCounter = 0

    sns.set_style("white")
    for i, y in enumerate(a1):
        for j, x in enumerate(a2):
            tempPred = predictions[predIdxCounter]
            argMax = np.argmax(tempPred)
            current_value = tempPred[argMax]
            if argMax == 1:
                utempPred = tempPred[argMax]
            else:
                utempPred = -1 * np.abs(tempPred[argMax])
                # utempPred = tempPred[argMax]

            prev_value = fin_img[y:y+imgSize, x:x+imgSize]
            average_value = (prev_value + utempPred) / 2

            decay_matrix = create_centered_matrix(imgSize, average_value, decay_rate)

            fin_img[y:y+imgSize, x:x+imgSize] += decay_matrix

            predIdxCounter += 1


    fig, axs = plt.subplots(1, 1, figsize=(6, 6), dpi=200)

    # Plot normalized interaction map
    im_norm = axs.imshow(fin_img, cmap='RdYlGn', vmax=1, vmin=-1)
    fig.colorbar(im_norm, ax=axs, fraction=0.02, pad=0.09)
    plt.xlabel(xp)
    plt.ylabel(yp)

    if not os.path.exists(opf):
        os.makedirs(opf)

    save_path = f'{opf}/{pid}_norm_intr_map.png'
    plt.savefig(save_path)

    plt.close(fig)

def process(ip_file_path, org, optype, optp, stride):
    
    op_file_path = f'./Figure/{org}/{optype}/Stride-{stride}/{optp}'  # you can set oytput Figure path here
    df = pd.read_csv(ip_file_path)

    # Add tqdm to iterate through the dataframe rows
#     for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing interactions", unit="PPI", leave=True):
    for index, row in df.iterrows():
        pid = f"{index}_{df.iloc[index]['Protein_1']}_{df.iloc[index]['Protein_2']}"
        pid = pid.replace('.', '-')
        yp = pid.split('_')[1]
        xp = pid.split('_')[2]

        seq1 = df.iloc[index]['Protein_seq1']
        seq2 = df.iloc[index]['Protein_seq2']

        imgSize = 128
        height = len(seq1)
        width = len(seq2)

        predictions = eval(df.iloc[index]['resultant_scores'])

        pred_img_dims = eval(df.iloc[index]['img_dim'])

        normalized_predictions = normalize(predictions)

        decay_rate = 0.1
        norm = True
        InteractionMapPlotComplete(pid, xp, yp, seq1, seq2, normalized_predictions, pred_img_dims, height, width, imgSize, stride, op_file_path, norm, decay_rate)
#         if index==5: break
    return index

org='Yeast'
opf='PROTEIN'
fpath=f'./ResultFiles/{org}/'  # path to predicted score Files
fpath='./currTestSCnegP16'
files = os.listdir(fpath)


csv_files = [inpf for inpf in files if inpf.endswith('.csv')]

for inpf in tqdm(csv_files, desc="Processing files", unit="file"):
    optp, stride = extract_file_info(inpf)  # Extract file info
    rc = process(f'{fpath}{inpf}', org, opf, optp, int(stride))
    print(f'{org}-{optp}-{stride}, {rc}, File: {inpf}')
import argparse
from Utils import *

def main():
    project_base_dir = "../Code/"
    parser = argparse.ArgumentParser(description="DensePPI-2 Prediction for FASTALike file using PAM120-SC Model")

    parser.add_argument('--input',
                        type=str,
                        help="Input file name with location")

    parser.add_argument('--output',
                        type=str,
                        help="Output file base path location")
    
    parser.add_argument('--stride',
                        type=int,
                        help="Stride in int format, default=64")
    
    parser.add_argument('--interaction_type',
                        type=str,
                        help="positive/negative")
    
    parser.add_argument('--device',
                        type=str,
                        help="cuda/cpu")
    
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    stride = args.stride
    interaction_type = args.interaction_type
    device = None
    if (args.device == "cuda" or args.device == "cuda:0" or args.device == "cuda:1"): 
        device = torch.device(args.device if torch.cuda.is_available() else "cpu") 
    else :
        device = torch.device(args.device) 
    print("Using device : ", device)

    set_global_stride(stride)
    model_saved_loc = project_base_dir + "saved_models/complete_epoch_010_metric_0.91918.pth.tar"
    intermediate_dir_name = interaction_type + "_result_sc_pam120_stride" + str(stride)
    outputSaveBasePath = output_file
    test_img_parent_loc = outputSaveBasePath + "testImagesOrig/"
    test_img_sub_loc = outputSaveBasePath + "testImagesSub/test_sub/"
    test_sub_img_dict = outputSaveBasePath + "test_image_dict.json"
    test_sub_img_prefix = "test"

    createLocationIfNotExists(outputSaveBasePath)
    createLocationIfNotExists(test_img_parent_loc)
    createLocationIfNotExists(test_img_sub_loc)
    substituton_matrix_init(outputSaveBasePath, 'pam', 120)

    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')

    prot2seqdict = {}
    intProtA = []
    intProtB = []
    counter = 1
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        if (counter % 2 == 1) :
            intProtA.append(name)
        else:
            intProtB.append(name)
        counter += 1    
        prot2seqdict[name] = trim_to_first_integer(sequence)

    df = pd.DataFrame(columns=['Index', 'Protein_1', 'Protein_2', 'Protein_seq1', 'Protein_seq2'])
    count = 1
    for i in range(len(intProtA)):
        li = list()
        li.append(count)
        li.append(intProtA[i])
        li.append(intProtB[i])
        li.append(prot2seqdict[intProtA[i]])
        li.append(prot2seqdict[intProtB[i]]) 
        df = df._append(pd.Series(li, index=df.columns), ignore_index=True)
        count+=1

    dict_out_loc = outputSaveBasePath + intermediate_dir_name + "_seq_dict.json"

    with open(dict_out_loc,"w") as outfile:
        json.dump(prot2seqdict, outfile)

    maxAllowedSequenceLength = np.inf
    print("======================CONSTRAINTS ON DATA======================")
    print("Maximum allowed sequence length : ", maxAllowedSequenceLength)
    print("Minimum allowed sequence length : ", SIZE)
    print("===============================================================")
    print("Before cleaning .....")
    print("Total Interactions : ", len(df))

    mask = lambda x : checkChars(x['Protein_seq1'], x['Protein_seq2'])
    df = df.loc[df.apply(mask, axis = 1)]

    mask = (df['Protein_seq1'].str.len() >= SIZE) & (df['Protein_seq2'].str.len() >= SIZE) & (df['Protein_seq1'].str.len() < maxAllowedSequenceLength) & (df['Protein_seq2'].str.len() < maxAllowedSequenceLength)
    df = df.loc[mask]

    df.reset_index(drop=True, inplace=True)
    print("After cleaning .....")
    print("Total Interactions : ", len(df))

    for it in range(len(df)):
        save_one_image_1C(it, df.iloc[it, 1], df.iloc[it, 2], df.iloc[it, 3], df.iloc[it, 4], test_img_parent_loc)

    pos_test_img_list = os.listdir(test_img_parent_loc)
    args = []
    for img in pos_test_img_list:
        args.append(test_img_parent_loc + img)
        
    imgCounter = SubImageCounter()

    num_workers = multiprocessing.cpu_count()  

    with multiprocessing.Pool(processes = num_workers) as pool:
        for images in pool.map(handle_one_image, args):				#Synchronus but in parallel
            image_list = []
            for image in images:		
                image_list.append(imgCounter.subimage_counter)
                file_name = str(test_img_sub_loc + test_sub_img_prefix + "_sub_" + str(imgCounter.subimage_counter) + ".png")
                plt.imsave(file_name, image)
                imgCounter.subimage_counter += 1
            imgCounter.image_dict[imgCounter.image_counter] = image_list
            imgCounter.image_counter += 1

    with open(test_sub_img_dict,"w") as outfile:
        json.dump(imgCounter.image_dict, outfile)

    test_sub_images_parent_loc = os.path.join(os.path.abspath(test_img_sub_loc), '..')
    test_data = datasets.ImageFolder(test_sub_images_parent_loc, transform=transformations)
    classLabel = 1 if interaction_type == "positive" else 0
    if (len(test_data.classes) == 1): #considering only negative
        test_data.samples = [(d, classLabel) for d, _ in test_data.samples]
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=bs)
    model = PPIModel()
    
    model = model.to(device=device)
    checkpoint = torch.load(model_saved_loc, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

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

    orig_labels, predictions, raw_preds = np.array(gr_tr), np.array(pred), np.array(raw_pred)

    generate_metrices_from_dict_single_type(orig_labels, predictions, raw_preds, test_sub_img_dict, outputSaveBasePath)

    f = open(outputSaveBasePath + "prediction_result_dict.json")
    p_data = json.load(f)
    f.close()

    args = []
    for it in range(len(df)):
        args.append(df.iloc[it, 1] + '_' + df.iloc[it, 2] + '.png')

    args = np.array(args)
    out_df = pd.DataFrame(columns=['Index', 'Protein_1', 'Protein_2', 
                'Protein_seq1', 'Protein_seq2', 'img_dim', 'resultant_scores'])

    with open(outputSaveBasePath + "input_output_means.pkl", 'rb') as f:
        _, fin_op, _ = pickle.load(f)

    for i in range(len(p_data)):
        protein = pos_test_img_list[i]
        im = Image.open(test_img_parent_loc + protein)
        im = np.array(im)
        height, width, _ = im.shape
        imgDim = [len(list(range(0, height-SIZE+STRIDE, STRIDE))), len(list(range(0, width-SIZE+STRIDE, STRIDE)))]
        str_to_end_with = protein.split('_', 1)[1]
        idx = np.where(args == str_to_end_with)[0][0]
        new_row = {
            'Index' : i+1,
            'Protein_1' : df.iloc[idx, 1],
            'Protein_2' : df.iloc[idx, 2],
            'Protein_seq1' : df.iloc[idx, 3],
            'Protein_seq2' : df.iloc[idx, 4],
            'img_dim' : imgDim,
            'resultant_scores' : p_data[str(i + 1)],
            'orig_pred' : fin_op[i]
        }
        out_df = out_df._append(pd.Series(new_row), ignore_index=True)

    out_df.to_csv(outputSaveBasePath + intermediate_dir_name + ".csv", index=False)

if __name__ == '__main__':
    main()
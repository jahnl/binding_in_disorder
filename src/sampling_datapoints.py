# create and save new datapoints based on the residue-wise oversampling process
# --> avoid long computation times during training

import numpy as np
import h5py
from Bio import SeqIO
import re



def read_labels(fold, oversampling):
    with open(f'../dataset/folds/CV_fold_{fold}_labels_{oversampling}.txt') as handle:
        records = SeqIO.parse(handle, "fasta")
        labels = dict()
        for record in records:
            # re-format input information to 3 sequences in a list per protein in dict labels{}
            # additionally record description for data points created via oversampling of residues (*)
            if '*' in record.id:
                seqs = list()
                seqs.append(record.seq[:int(len(record.seq) / 3)])
                seqs.append(record.seq[int(len(record.seq) / 3):2 * int(len(record.seq) / 3)])
                seqs.append(record.seq[2 * int(len(record.seq) / 3):])
                seqs.append(record.description.split('\t')[1][:-1])
                labels[record.id] = seqs
                """
                if record.id == 'E2IHW6*':
                    print(record.id, labels[record.id], record)
                """
    return labels


def get_ML_data(labels, embeddings, mode):
    input = list()
    for id in labels.keys():
        if mode == 'all':
            conf_feature = str(labels[id][1])
            conf_feature = list(conf_feature.replace('-', '0').replace('D', '1'))
            conf_feature = np.array(conf_feature, dtype=float)

            print(f'building new embedding for data points of {id}...')
            indices = labels[id][3].split(', ')
            emb = np.array(embeddings[id[:-1]][int(indices[0])], dtype=float)
            for i in indices[1:]:
                emb = np.row_stack((emb, embeddings[id[:-1]][int(i)]))
            emb_with_conf = np.column_stack((emb, conf_feature))

            input.append(emb_with_conf)

        elif mode == 'disorder_only':
            # not implemented yet!
            bool_list = [False if x == '-' else True for x in list(labels[id][2])]
            input.append(embeddings[id][bool_list])

    return input


if __name__ == '__main__':
    # apply cross-validation and oversampling on training dataset
    oversampling = 'multiclass_residues'
    #CV_splits.split(oversampling)

    mode = 'all'  # disorder_only or all

    # read input embeddings
    embeddings_in = '../dataset/train_set.h5'
    embeddings = dict()
    with h5py.File(embeddings_in, 'r') as f:
        for key, embedding in f.items():
            original_id = embedding.attrs['original_id']
            embeddings[original_id] = np.array(embedding)
    # now {IDs: embeddings} are written in the embeddings dictionary

    # iterate over folds
    for fold in range(5):
        print("Fold: " + str(fold))
        # read target data y and disorder information
        # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
        labels = read_labels(fold, oversampling)

        # create the input and target data exactly how it's fed into the ML model
        # and add the confounding feature of disorder to the embeddings
        # and save new embeddings to file
        new_embs = get_ML_data(labels, embeddings, mode)
        np.save(file=f'../dataset/folds/new_datapoints_{oversampling}_fold_{fold}.npy', arr=new_embs)

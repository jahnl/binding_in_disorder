"""
starting point for predictor development:
simple model, binary prediction of binding/non-binding,
given embeddings + confounding feature: disorder
labels: anything vs '_' in 4th line of labels per protein
no oversampling yet
"""


from src import CV_splits
import numpy as np
import h5py
from Bio import SeqIO
import re


def read_labels(fold):
    with open(f'../dataset/folds/CV_fold_{fold}_labels.txt') as handle:
        records = SeqIO.parse(handle, "fasta")
        labels = dict()
        for record in records:
            # re-format input information to 3 sequences in a list per protein in dict labels{}
            seqs = list()
            seqs.append(record.seq[:int(len(record.seq) / 3)])
            seqs.append(record.seq[int(len(record.seq) / 3):2 * int(len(record.seq) / 3)])
            seqs.append(record.seq[2 * int(len(record.seq) / 3):])
            labels[record.id] = seqs
            if record.id == 'Q98157':
                print(record.id, labels[record.id])
    return labels


def get_ML_data(labels, embeddings):
    input = list()
    target = list()
    for id in labels.keys():
        conf_feature = str(labels[id][1])
        conf_feature = list(conf_feature.replace('-', '0').replace('D', '1'))
        conf_feature = np.array(conf_feature, dtype=float)
        emb_with_conf = np.column_stack((embeddings[id], conf_feature))
        input.append(emb_with_conf)
        # for target: 0 = non-binding, 1 = binding, 2 = not in disordered region
        binding = str(labels[id][2])
        binding = binding.replace('-', '2').replace('_', '0')
        binding = list(re.sub(r'P|N|O|X|Y|Z|A', '1', binding))
        binding = np.array(binding, dtype=float)
        target.append(binding)
        if id == 'Q98157':
            print(conf_feature)
            print(emb_with_conf.shape)
            print(binding)
    return input, target


if __name__ == '__main__':
    # apply cross-validation on training dataset
    #CV_splits.split()

    # read input embeddings
    embeddings_in = '../dataset/train_set.h5'
    embeddings = dict()
    with h5py.File(embeddings_in, 'r') as f:
        for key, embedding in f.items():
            original_id = embedding.attrs['original_id']
            embeddings[original_id] = np.array(embedding)
    # now {IDs: embeddings} are written in the embeddings dictionary

    # iterate over folds
    for fold in [0]:     # range(5):    TODO: use required fold, (use config file later on)
        print("Fold: " + str(fold))
        # for training use all training IDs except for the ones in the current fold.
        # for validation use the training IDs in the current fold

        # read target data y and disorder information
        # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
        val_labels = read_labels(fold)
        train_labels = {}
        for train_fold in range(5):     # TODO: link number of files to config file
            if train_fold != fold:
                train_labels.update(read_labels(train_fold))
        print(len(val_labels), len(train_labels))

        # create the input and target data exactly how it's fed into the ML model
        # and add the confounding feature of disorder to the embeddings
        this_fold_input, this_fold_target = get_ML_data(train_labels, embeddings)
        this_fold_val_input, this_fold_val_target = get_ML_data(val_labels, embeddings)

        # instantiate the dataset
        # TODO next



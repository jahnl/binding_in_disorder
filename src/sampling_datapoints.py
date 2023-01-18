"""
create and save new data points based on the residue-wise oversampling process
--> avoid long computation times during training
special case: create AAindex representation for the baseline model instead
"""

import numpy as np
import pandas as pd
import h5py
from Bio import SeqIO
from os.path import exists


def read_labels(fold, oversampling, dataset_dir):
    with open(f'{dataset_dir}folds/CV_fold_{fold}_labels_{oversampling}.txt', 'r') as handle:
        records = SeqIO.parse(handle, "fasta")
        labels = dict()
        for record in records:
            # re-format input information to 3 sequences in a list per protein in dict labels{}
            # additionally record description for data points created via oversampling of residues (*/$)
            if '*' in record.id or '$' in record.id:
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


def read_all_labels(fold, oversampling, dataset_dir):
    file = f'{dataset_dir}test_set_input.txt' if fold is None else f'{dataset_dir}folds/CV_fold_{fold}_labels_{oversampling}.txt'
    with open(file, 'r') as handle:
        records = SeqIO.parse(handle, "fasta")
        labels = dict()
        for record in records:
            # re-format input information to 3 sequences in a list per protein in dict labels{}
            seqs = list()
            seqs.append(record.seq[:int(len(record.seq) / 3)])
            seqs.append(record.seq[int(len(record.seq) / 3):2 * int(len(record.seq) / 3)])
            seqs.append(record.seq[2 * int(len(record.seq) / 3):])
            labels[record.id] = seqs
            """
            if record.id == 'E2IHW6*':
                print(record.id, labels[record.id], record)
            """
    return labels


def get_ML_data(labels, embeddings, mode, database):
    input = list()
    for id in labels.keys():
        if mode == 'all':
            conf_feature = str(labels[id][1])
            conf_feature = list(conf_feature.replace('-', '0').replace('D', '1'))
            conf_feature = np.array(conf_feature, dtype=float)

            print(f'building new embedding for data points of {id}...')
            indices = labels[id][3].split(', ')
            name = id[:-1] if database == 'disprot' else id[1:]
            emb = np.array(embeddings[name][int(indices[0])], dtype=float)
            for i in indices[1:]:
                emb = np.row_stack((emb, embeddings[name][int(i)]))
            emb_with_conf = np.column_stack((emb, conf_feature))

            input.append(emb_with_conf)

        elif mode == 'disorder_only':
            # this only filters for disordered residues,
            # picking the correct residues from the index list not implemented yet! TODO
            bool_list = [False if x == '-' else True for x in list(labels[id][2])]
            input.append(embeddings[id][bool_list])

        # else: do nothing

    return input


def AAindex_rep(labels, aaindex):
    all_reps = dict()
    for id in labels.keys():
        print(id)
        rep = None
        for residue in labels[id][0]:
            if rep is None:
                try:
                    rep = np.array(aaindex[residue])
                except KeyError:
                    rep = np.zeros(shape=566)
                    print(f"represented AA {residue} in {id} with vector of Zeros")
            else:
                try:
                    rep = np.row_stack((rep, np.array(aaindex[residue])))
                except KeyError:
                    rep = np.row_stack((rep, np.zeros(shape=566)))
                    print(f"represented AA {residue} in {id} with vector of Zeros")
        all_reps[id] = rep
    return all_reps


def sample_datapoints(train_embeddings: str, dataset_dir: str, database: str, oversampling: str = 'binary_residues',
                      mode: str = 'all', n_splits: int = 5):
    """
    create new embeddings for the residue-wise oversampled datapoints
    :param database: mobidb or disprot annotation format
    :param dataset_dir: directory where the dataset files are stored
    :param train_embeddings: path to the embedding file of the train set datapoints
    :param oversampling: 'binary_residues', 'binary_residues_disorder' or 'multiclass_residues'
    :param mode: either 'disorder_only' or 'all'
    :param n_splits: number of cross-validation splits
    """

    # special case: creation of protein representation from AAindex1
    if train_embeddings == '':
        aaindex = pd.read_csv("../dataset/AAindex1/aaindex1.csv")
        for fold in range(n_splits):
            print("Fold: " + str(fold))
            # read target data y and disorder information
            # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
            labels = read_all_labels(fold, oversampling, dataset_dir)
            # create the input and target data exactly how it's fed into the ML model
            # and save new representations to file
            representation = AAindex_rep(labels, aaindex)
            np.save(file=f'{dataset_dir}folds/AAindex_representation_{oversampling}_fold_{fold}.npy', arr=representation)

        if not exists(f'{dataset_dir}AAindex_representation_test.npy'):
            print("Test set")
            labels = read_all_labels(None, oversampling, dataset_dir)
            representation = AAindex_rep(labels, aaindex)
            np.save(file=f'{dataset_dir}AAindex_representation_test.npy', arr=representation)
        return


    # else:
    # read input embeddings
    embeddings = dict()
    with h5py.File(train_embeddings, 'r') as f:
        for key, embedding in f.items():
            original_id = embedding.attrs['original_id']
            embeddings[original_id] = np.array(embedding)
    # now {IDs: embeddings} are written in the embeddings dictionary

    # iterate over folds
    for fold in range(n_splits):
        print("Fold: " + str(fold))
        # read target data y and disorder information
        # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
        labels = read_labels(fold, oversampling, dataset_dir)

        # create the input and target data exactly how it's fed into the ML model
        # and add the confounding feature of disorder to the embeddings
        # and save new embeddings to file
        new_embs = get_ML_data(labels, embeddings, mode, database)
        np.save(file=f'{dataset_dir}folds/new_datapoints_{oversampling}_fold_{fold}.npy', arr=new_embs)

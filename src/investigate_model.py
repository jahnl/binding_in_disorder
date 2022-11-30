"""
investigation of a newly trained model
includes determination of an optimal cutoff and final prediction, with or without post-processing
"""

import numpy as np
import h5py
from Bio import SeqIO
import re
import torch.tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import nn
from os.path import exists


def read_labels(fold, oversampling, dataset_dir):
    if fold is None:  # --> test set
        file_name = f'{dataset_dir}test_set_input.txt'
    else:
        if oversampling is None:  # no oversampling on validation set! (or mode with no oversampling)
            file_name = f'{dataset_dir}folds/CV_fold_{fold}_labels.txt'
            if not exists(file_name):
                file_name = f'{dataset_dir}folds/CV_fold_{fold}_labels_None.txt'
        else:
            file_name = f'{dataset_dir}folds/CV_fold_{fold}_labels_{oversampling}.txt'
    with open(file_name, 'r') as handle:
        records = SeqIO.parse(handle, "fasta")
        labels = dict()
        for record in records:
            # re-format input information to 3 sequences in a list per protein in dict labels{}
            seqs = list()
            seqs.append(record.seq[:int(len(record.seq) / 3)])
            seqs.append(record.seq[int(len(record.seq) / 3):2 * int(len(record.seq) / 3)])
            seqs.append(record.seq[2 * int(len(record.seq) / 3):])
            labels[record.id] = seqs
    return labels


def get_ML_data(labels, embeddings, architecture, mode, multilabel, new_datapoints):
    input = list()
    target = list()
    datapoint_counter = 0
    disorder = []
    for id in labels.keys():
        if mode == 'all' or multilabel:
            conf_feature = str(labels[id][1])
            conf_feature = list(conf_feature.replace('-', '0').replace('D', '1'))
            conf_feature = np.array(conf_feature, dtype=float)
            disorder.append(conf_feature)
            if '*' not in id:
                emb_with_conf = np.column_stack((embeddings[id], conf_feature))
            else:  # data points created by residue-wise oversampling
                # use pre-computed embedding
                emb_with_conf = new_datapoints[datapoint_counter]
                datapoint_counter += 1
                if emb_with_conf.shape[0] != len(labels[id][1]):  # sanity check
                    raise ValueError(f'Wrong match between label and embedding. Label of {id} has length '
                                     f'{len(labels[id][1])}, emb has shape {emb_with_conf.shape}')

            input.append(emb_with_conf)
        elif mode == 'disorder_only':
            if architecture == 'FNN':
                bool_list = [False if x == '-' else True for x in list(labels[id][2])]
                input.append(embeddings[id][bool_list])
                disorder.append([1]*len(labels[id][2]))
                #disorder.append(list('1'*len(labels[id][2])))
            else:   # CNN
                # separate regions of the same protein from each other!
                bool_list = [False if x == '-' else True for x in list(labels[id][2])]
                starts = []
                stops = []
                current = bool_list[0]
                diso_start = 0
                for i, diso_residue in enumerate(bool_list):
                    if diso_residue != current and diso_residue:  # order to disorder conformation change
                        diso_start = i
                    elif diso_residue != current and not diso_residue:  # disorder to order conformation change
                        input.append(embeddings[id][diso_start:i])
                        disorder.append([1] * (i-diso_start))
                        #disorder.append(list('1' * (i-diso_start)))
                        starts.append(diso_start)
                        stops.append(i)
                        # print(f'{id}: region from {diso_start} to {i}')
                    current = diso_residue
                # disorder at final residue?
                if current:
                    input.append(embeddings[id][diso_start:])
                    disorder.append([1] * (len(bool_list)-diso_start))
                    #disorder.append(list('1' * (len(bool_list)-diso_start)))
                    starts.append(diso_start)
                    stops.append(len(bool_list))
                    # print(f'{id}: region from {diso_start} to {stops[-1]}')

                # binding data for CNN + disorder_only
                for i, s in enumerate(starts):
                    binding = str(labels[id][2][starts[i]:stops[i]])
                    binding = re.sub('_', '0', binding)
                    binding = list(re.sub(r'B|P|N|O|X|Y|Z|A', '1', binding))
                    binding = np.array(binding, dtype=float)
                    target.append(binding)

        if not multilabel:
            if not (architecture == 'CNN' and mode == 'disorder_only'):
                # for target: 0 = non-binding or not in disordered region, 1 = binding
                binding = str(labels[id][2])
                if mode == 'all':
                    binding = re.sub(r'-|_', '0', binding)
                elif mode == 'disorder_only':
                    binding = binding.replace('-', '').replace('_', '0')
                binding = list(re.sub(r'B|P|N|O|X|Y|Z|A', '1', binding))
                binding = np.array(binding, dtype=float)
                target.append(binding)
        else:
            # for target: 0 = non-binding or not in disordered region, 1 = binding; 3-dimensions per residue
            binding = str(labels[id][2])
            binding_encoded = [[], [], []]
            binding_encoded[0] = list(re.sub(r'P|X|Y|A', '1', re.sub(r'-|_|N|O|Z', '0', binding)))  # protein-binding?
            binding_encoded[1] = list(
                re.sub(r'N|X|Z|A', '1', re.sub(r'-|_|P|O|Y', '0', binding)))  # nucleic-acid-binding?
            binding_encoded[2] = list(re.sub(r'O|Y|Z|A', '1', re.sub(r'-|_|P|N|X', '0', binding)))  # other-binding?
            target.append(np.array(binding_encoded, dtype=float).T)

    return input, target, disorder


# build the dataset
class BindingDataset(Dataset):
    def __init__(self, embeddings, binding_labels, disorder_labels, architecture: str):
        self.inputs = embeddings
        self.labels = binding_labels
        self.disorder = disorder_labels
        if architecture in ["CNN", "FNN"]:
            self.architecture = architecture
        else:
            raise ValueError('architecture must be "FNN" or "CNN"')

    def __len__(self):
        if self.architecture == 'CNN':
            # this time the batch size = number of proteins = number of datapoints for the dataloader
            return len(self.labels)
        else:  # FNN
            return sum([len(protein) for protein in self.labels])

    def number_residues(self):
        return sum([len(protein) for protein in self.labels])

    def number_diso_residues(self):
        return sum([sum(d) for d in self.disorder])

    def __getitem__(self, index):
        if self.architecture == 'CNN':
            # 3-dimensional input must be provided to conv1d, so proteins must be organised in batches
            try:
                return torch.tensor(self.inputs[index]).float(), torch.tensor(self.labels[index], dtype=torch.long), \
                       torch.tensor(self.disorder[index], dtype=torch.long)
            except IndexError:
                return None
        else:  # FNN
            k = 0  # k is the current protein index, index gets transformed to the position in the sequence
            protein_length = len(self.labels[k])
            while index >= protein_length:
                index = index - protein_length
                k += 1
                protein_length = len(self.labels[k])
            return torch.tensor(self.inputs[k][index]).float(), torch.tensor(self.labels[k][index]), \
                torch.tensor(self.disorder[k][index])


class CNN(nn.Module):
    def __init__(self, n_layers: int, kernel_size: int, dropout: float, input_size: int):
        super().__init__()
        self.n_layers = n_layers
        padding = int((kernel_size - 1) / 2)
        if self.n_layers == 2:
            # version 0: 2 C layers
            self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=kernel_size, padding=padding)
            # --> out: (32, proteins_length)
            self.dropout = nn.Dropout(p=dropout)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=kernel_size, padding=padding)
            # --> out: (1, protein_length)
        elif self.n_layers == 5:
            # version 1: 5 C layers
            self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=kernel_size, padding=padding)
            self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=kernel_size, padding=padding)
            self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=kernel_size, padding=padding)
            self.conv5 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=kernel_size, padding=padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout)
            # --> out: (1, protein_length)
        elif self.n_layers == 8:
            self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=kernel_size, padding=padding)
            self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=kernel_size, padding=padding)
            self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=kernel_size, padding=padding)
            self.conv5 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=kernel_size, padding=padding)
            self.conv6 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=kernel_size, padding=padding)
            self.conv7 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=kernel_size, padding=padding)
            self.conv8 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=kernel_size, padding=padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout)
            # --> out: (1, protein_length)

    def forward(self, input):
        if self.n_layers == 2:
            # version 0: 2 C layers
            x = self.conv1(input.transpose(1, 2).contiguous())
            x = self.dropout(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = x + 2
        elif self.n_layers == 5:
            # version 1: 5 C layers
            x = self.conv1(input.transpose(1, 2).contiguous())
            x = self.dropout(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.relu(x)
            x = self.conv5(x)
            x += 3
        elif self.n_layers == 8:
            # version 2: 8 C layers
            x = self.conv1(input.transpose(1, 2).contiguous())
            x = self.dropout(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.relu(x)
            x = self.conv5(x)
            x = self.relu(x)
            x = self.conv6(x)
            x = self.relu(x)
            x = self.conv7(x)
            x = self.relu(x)
            x = self.conv8(x)
            x += 4
        return x


class FNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float, multilabel: bool):
        super(FNN, self).__init__()
        self.input_layer = nn.Linear(input_size, input_size)
        self.hidden_layer = nn.Linear(input_size, int(input_size / 2))
        self.output_layer = nn.Linear(int(input_size / 2), output_size)
        self.dropout = nn.Dropout(dropout)
        self.multilabel = multilabel

    def forward(self, input):
        x = F.relu(self.input_layer(input))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer(x))
        x = self.dropout(x)
        if self.multilabel:
            output = torch.sigmoid(self.output_layer(x))
        else:
            output = self.output_layer(x)  # rather without sigmoid to apply BCEWithLogitsLoss later
        return output


def transform_output(p, n, o):
    binding_code = p * 100 + n * 10 + o
    transformation = {0: '_',
                      100: 'P',
                      10: 'N',
                      1: 'O',
                      110: 'X',
                      101: 'Y',
                      11: 'Z',
                      111: 'A'}
    return transformation[binding_code]


def try_cutoffs(model_name: str, dataset_dir: str, embeddings, mode: str = 'all', multilabel: bool = False,
                n_splits: int = 5,
                architecture: str = 'FNN', n_layers: int = 0, kernel_size: int = 5, batch_size: int = 512, cutoff_percent_min: int = 0,
                cutoff_percent_max: int = 100, step_percent: int = 5, dropout: float = 0.3):
    def criterion(loss_func, prediction, label):  # sum over all classification heads
        losses = 0
        prediction = prediction.T
        label = label.T
        for i, _ in enumerate(prediction):  # for each class (-> 1-dimensional loss)
            losses += loss_func(prediction[i], label[i])
        return losses

    def test_performance(dataset, model, loss_function, device, output, multilabel, batch_size, cutoff_percent_min,
                         cutoff_percent_max, step_percent):
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        size = dataset.number_residues()
        diso_size = dataset.number_diso_residues()
        # print(size)
        model.eval()
        with torch.no_grad():
            # try out different cutoffs
            # multiplication works around floating point precision issue
            for cutoff in (0.01 * np.arange(cutoff_percent_min, cutoff_percent_max, step=step_percent)):
                if multilabel:
                    test_loss = 0
                    p_correct, p_tp, p_fp, p_tn, p_fn = 0, 0, 0, 0, 0
                    n_correct, n_tp, n_fp, n_tn, n_fn = 0, 0, 0, 0, 0
                    o_correct, o_tp, o_fp, o_tn, o_fn = 0, 0, 0, 0, 0
                    for input, label in test_loader:
                        input, label = input.to(device), label.to(device)
                        prediction = model(input)
                        test_loss += criterion(loss_function, prediction, label.to(torch.float32)).item()
                        # apply activation function to prediction to get classification and transpose matrices
                        prediction_max = (prediction > cutoff).T
                        label = label.T

                        # metrics
                        p_correct += (prediction_max[0] == label[0]).type(torch.float).sum().item()
                        p_tp += (prediction_max[0] == label[0])[label[0] == 1].type(torch.float).sum().item()
                        p_fp += (prediction_max[0] != label[0])[label[0] == 0].type(torch.float).sum().item()
                        p_tn += (prediction_max[0] == label[0])[label[0] == 0].type(torch.float).sum().item()
                        p_fn += (prediction_max[0] != label[0])[label[0] == 1].type(torch.float).sum().item()

                        n_correct += (prediction_max[1] == label[1]).type(torch.float).sum().item()
                        n_tp += (prediction_max[1] == label[1])[label[1] == 1].type(torch.float).sum().item()
                        n_fp += (prediction_max[1] != label[1])[label[1] == 0].type(torch.float).sum().item()
                        n_tn += (prediction_max[1] == label[1])[label[1] == 0].type(torch.float).sum().item()
                        n_fn += (prediction_max[1] != label[1])[label[1] == 1].type(torch.float).sum().item()

                        o_correct += (prediction_max[2] == label[2]).type(torch.float).sum().item()
                        o_tp += (prediction_max[2] == label[2])[label[2] == 1].type(torch.float).sum().item()
                        o_fp += (prediction_max[2] != label[2])[label[2] == 0].type(torch.float).sum().item()
                        o_tn += (prediction_max[2] == label[2])[label[2] == 0].type(torch.float).sum().item()
                        o_fn += (prediction_max[2] != label[2])[label[2] == 1].type(torch.float).sum().item()

                    test_loss /= int(size / batch_size)
                    p_correct /= size
                    n_correct /= size
                    o_correct /= size
                    # class-wise printing - maybe different cutoffs required
                    try:
                        print(
                            f"cutoff {cutoff}\tProtein-binding:\tAccuracy: {(100 * p_correct):>0.1f}%, Sensitivity: {(100 * (p_tp / (p_tp + p_fn))): >0.1f}%, Precision: {(100 * (p_tp / (p_tp + p_fp))): >0.1f}%, Avg loss: {test_loss:>8f}")
                        output.write('\t'.join(
                            [str(fold), str(round(test_loss, 6)), str(cutoff),
                             str(round(100 * p_correct, 1)), str(round(100 * (p_tp / (p_tp + p_fp)), 1)),
                             str(round(100 * (p_tp / (p_tp + p_fn)), 1)),
                             str(p_tp), str(p_fp), str(p_tn), str(p_fn)]) + '\t')
                    except ZeroDivisionError:
                        print(
                            f"cutoff {cutoff}\tProtein-binding:\tAccuracy: {(100 * p_correct):>0.1f}%, Sensitivity: 0.0%, Precision: 0.0%, Avg loss: {test_loss:>8f}")
                        output.write('\t'.join(
                            [str(fold), str(round(test_loss, 6)), str(cutoff),
                             str(round(100 * p_correct, 1)), '0.0', '0.0',
                             str(p_tp), str(p_fp), str(p_tn), str(p_fn)]) + '\t')
                    try:
                        print(
                            f"cutoff {cutoff}\tNuc-binding:\tAccuracy: {(100 * n_correct):>0.1f}%, Sensitivity: {(100 * (n_tp / (n_tp + n_fn))): >0.1f}%, Precision: {(100 * (n_tp / (n_tp + n_fp))): >0.1f}%, Avg loss: {test_loss:>8f}")
                        output.write('\t'.join(
                            [str(round(100 * n_correct, 1)), str(round(100 * (n_tp / (n_tp + n_fp)), 1)),
                             str(round(100 * (n_tp / (n_tp + n_fn)), 1)),
                             str(n_tp), str(n_fp), str(n_tn), str(n_fn)]) + '\t')
                    except ZeroDivisionError:
                        print(
                            f"cutoff {cutoff}\tNuc-binding:\tAccuracy: {(100 * n_correct):>0.1f}%, Sensitivity: 0.0%, Precision: 0.0%, Avg loss: {test_loss:>8f}")
                        output.write('\t'.join(
                            [str(round(100 * n_correct, 1)), '0.0', '0.0',
                             str(n_tp), str(n_fp), str(n_tn), str(n_fn)]) + '\t')
                    try:
                        print(
                            f"cutoff {cutoff}\tOther-binding:\tAccuracy: {(100 * o_correct):>0.1f}%, Sensitivity: {(100 * (o_tp / (o_tp + o_fn))): >0.1f}%, Precision: {(100 * (o_tp / (o_tp + o_fp))): >0.1f}%, Avg loss: {test_loss:>8f}\n")
                        output.write('\t'.join(
                            [str(round(100 * o_correct, 1)), str(round(100 * (o_tp / (o_tp + o_fp)), 1)),
                             str(round(100 * (o_tp / (o_tp + o_fn)), 1)),
                             str(o_tp), str(o_fp), str(o_tn), str(o_fn)]) + '\n')
                    except ZeroDivisionError:
                        print(
                            f"cutoff {cutoff}\tOther-binding:\tAccuracy: {(100 * o_correct):>0.1f}%, Sensitivity: 0.0%, Precision: 0.0%, Avg loss: {test_loss:>8f}\n")
                        output.write('\t'.join(
                            [str(round(100 * o_correct, 1)), '0.0', '0.0',
                             str(o_tp), str(o_fp), str(o_tn), str(o_fn)]) + '\n')

                else:  # not multilabel
                    test_loss, correct, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0
                    diso_loss, diso_correct, diso_tp, diso_fp, diso_tn, diso_fn = 0, 0, 0, 0, 0, 0
                    for input, label, disorder in test_loader:
                        input, label, disorder = input.to(device), label[:, None].to(device), \
                                                 disorder[:, None].to(device)
                        prediction = model(input)
                        test_loss += loss_function(prediction, label.to(torch.float32)).item()
                        # apply activation function to prediction to get classification
                        prediction_act = torch.sigmoid(prediction)
                        prediction_max = prediction_act > cutoff
                        # metrics
                        correct += (prediction_max == label).type(torch.float).sum().item()
                        tp += (prediction_max == label)[label == 1].type(torch.float).sum().item()
                        fp += (prediction_max != label)[label == 0].type(torch.float).sum().item()
                        tn += (prediction_max == label)[label == 0].type(torch.float).sum().item()
                        fn += (prediction_max != label)[label == 1].type(torch.float).sum().item()
                        # metrics within disorder
                        diso_correct += (prediction_max == label)[disorder == 1].type(torch.float).sum().item()
                        mask_0 = (label == 0) & (disorder == 1)
                        mask_1 = (label == 1) & (disorder == 1)
                        diso_tp += (prediction_max == label)[mask_1].type(torch.float).sum().item()
                        diso_fp += (prediction_max != label)[mask_0].type(torch.float).sum().item()
                        diso_tn += (prediction_max == label)[mask_0].type(torch.float).sum().item()
                        diso_fn += (prediction_max != label)[mask_1].type(torch.float).sum().item()

                    test_loss /= int(size / batch_size)
                    correct /= size
                    diso_loss /= diso_size  # just to be comparable...
                    diso_correct /= diso_size
                    try:
                        print(
                            f"cutoff {cutoff}\tAccuracy: {(100 * correct):>0.1f}%, Sensitivity: {(100 * (tp / (tp + fn))): >0.1f}%, Precision: {(100 * (tp / (tp + fp))): >0.1f}%, Avg loss: {test_loss:>8f}, ")
                        output.write(
                            '\t'.join([str(fold), str(round(test_loss, 6)), str(cutoff), str(round(100 * correct, 1)),
                                       str(round(100 * (tp / (tp + fp)), 1)), str(round(100 * (tp / (tp + fn)), 1)),
                                       str(tp), str(fp), str(tn), str(fn)]) + '\t')
                    except ZeroDivisionError:
                        print(
                            f"cutoff {cutoff}\tAccuracy: {(100 * correct):>0.1f}%, Sensitivity: 0.0%, Precision: 0.0%, Avg loss: {test_loss:>8f}")
                        output.write('\t'.join(
                            [str(fold), str(round(test_loss, 6)), str(cutoff), str(round(100 * correct, 1)),
                             '0.0', '0.0', str(tp), str(fp), str(tn), str(fn)]) + '\t')

                    try:
                        print(
                            f"\t D_Accuracy: {(100 * diso_correct):>0.1f}%, D_N_Sens: {(100 * (diso_tn / (diso_tn + diso_fp))): >0.1f}%, D_N_Prec: {(100 * (diso_tn / (diso_tn + diso_fn))): >0.1f}%, D_Avg loss: {diso_loss:>8f}")
                        output.write(
                            '\t'.join([str(round(100 * diso_correct, 1)),
                                       str(round(100 * (diso_tn / (diso_tn + diso_fn)), 1)),
                                       str(round(100 * (diso_tn / (diso_tn + diso_fp)), 1)),
                                       str(round(diso_loss, 6)),
                                       str(diso_tp), str(diso_fp), str(diso_tn), str(diso_fn)]) + '\n')
                    except ZeroDivisionError:
                        print(
                            f"\t D_Accuracy: {(100 * diso_correct):>0.1f}%, D_N_Sens: 0.0%, D_N_Prec: 0.0%, D_Avg loss: {diso_loss:>8f}")
                        output.write(
                            '\t'.join([str(round(100 * diso_correct, 1)),
                                       '0.0', '0.0', str(round(diso_loss, 6)),
                                       str(diso_tp), str(diso_fp), str(diso_tn), str(diso_fn)]) + '\n')

    # iterate over folds
    with open(f"../results/logs/validation_{model_name}.txt", "w") as output_file:
        if multilabel:
            output_file.write('Fold\tAvg_Loss\tCutoff\tP_Acc\tP_Prec\tP_Rec\tP_TP\tP_FP\tP_TN\tP_FN\t'
                              'N_Acc\tN_Prec\tN_Rec\tN_TP\tN_FP\tN_TN\tN_FN\t'
                              'O_Acc\tO_Prec\tO_Rec\tO_TP\tO_FP\tO_TN\tO_FN\n')
        else:
            output_file.write('Fold\tAvg_Loss\tCutoff\tAcc\tPrec\tRec\tTP\tFP\tTN\tFN\t'
                              'D_Acc\tD_NPrec\tD_NRec\tD_Loss\tD_TP\tD_FP\tD_TN\tD_FN\n')

        for fold in range(n_splits):
            print("Fold: " + str(fold))
            # for validation use the training IDs in the current fold
            # read target data y and disorder information
            # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
            val_labels = read_labels(fold, None, dataset_dir)  # no oversampling on validation labels
            # create the input and target data exactly how it's fed into the ML model
            # and add the confounding feature of disorder to the embeddings
            this_fold_val_input, this_fold_val_target, this_fold_disorder = \
                get_ML_data(val_labels, embeddings, architecture, mode, multilabel, None)
            # instantiate the dataset
            validation_dataset = BindingDataset(this_fold_val_input, this_fold_val_target, this_fold_disorder,
                                                architecture)
            """
            # look at some data:
            for i in range(50, 54):
               input, label = training_dataset[i]
               print(f'Embedding input:\n {input}\nPrediction target:\n{label}\n\n')
            """

            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_size = 1024 if mode == 'disorder_only' else 1025
            output_size = 3 if multilabel else 1
            if architecture == 'FNN':
                model = FNN(input_size=input_size, output_size=output_size, dropout=dropout, multilabel=multilabel) \
                    .to(device)
            else:
                model = CNN(n_layers=n_layers, kernel_size=kernel_size, dropout=dropout, input_size=input_size).to(device)
                batch_size = 1  # batch size is always 1 (protein) if the model is a CNN
            model.load_state_dict(
                torch.load(f"../results/models/binding_regions_model_{model_name}_fold_{fold}.pth"))
            # test performance again, should be the same
            loss_function = nn.BCELoss() if multilabel else nn.BCEWithLogitsLoss()
            test_performance(validation_dataset, model, loss_function, device, output_file, multilabel, batch_size,
                             cutoff_percent_min, cutoff_percent_max, step_percent)


def predictCNN(embeddings, dataset_dir, cutoff, fold, model_name: str, n_layers, kernel_size, dropout, mode, test):
    output_name = f"../results/logs/predict_val_{model_name}_{fold}_{cutoff}.txt" if not test else \
        f"../results/logs/predict_val_{model_name}_{cutoff}_test.txt"
    with open(output_name, "w") as output_file:
        if not test:
            print("Fold: " + str(fold))
            # for validation use the training IDs in the current fold

            # read target data y and disorder information
            # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
            val_labels = read_labels(fold, None, dataset_dir)
        else:
            val_labels = read_labels(None, None, dataset_dir)

        ids = list(val_labels.keys())

        # create the input and target data exactly how it's fed into the ML model
        # and add the confounding feature of disorder to the embeddings
        this_fold_val_input, this_fold_val_target, _ = get_ML_data(val_labels, embeddings, 'CNN', mode, False, None)

        # instantiate the dataset
        validation_dataset = BindingDataset(this_fold_val_input, this_fold_val_target, 'CNN')

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CNN(n_layers, kernel_size, dropout).to(device)
        model.load_state_dict(
            torch.load(f"../results/models/binding_regions_model_{model_name}_fold_{fold}.pth"))
        # test performance again, should be the same

        test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)
        model.eval()
        with torch.no_grad():
            for i, (input, label) in enumerate(test_loader):
                input, label = input.to(device), label[None, :].to(device)
                prediction = model(input)
                # apply activation function to prediction to enable classification
                prediction_act = torch.sigmoid(prediction)
                prediction_max = prediction_act > cutoff

                output_file.write(
                    f'{ids[i]}\nlabels:\t{label}\nprediction_0:\t{prediction_act}\nprediction_1:\t{prediction_max}\n')


class Zone:
    def __init__(self, start: int, end: int, value: float, last: bool):
        # start incl, end excl.
        self.length = end - start
        self.value = value
        # identify specific zones in prediction:
        # pos_short: positive, len < 5, not at the end
        # neg_short: negative, len < 10, not at the start or end
        # pos_medium: positive, not short, len < 55
        # pos_long: positive, 55 <= len <= 240
        # pos_valid: positive, len > 240
        # neg_valid: negative, len >= 10
        if self.value == 0:
            if self.length < 10 and start != 0 and not last:
                self.type = "neg_short"
            else:
                self.type = "neg_valid"
        else:
            if self.length < 5 and not last:
                self.type = "pos_short"
                self.value = 0.0
            elif self.length < 55:
                self.type = "pos_medium"
            elif self.length < 241:
                self.type = "pos_long"
            else:
                self.type = "pos_valid"

    def get_value(self):
        return self.value

    def set_value(self, value: float):
        self.value = value

    def get_length(self):
        return self.length

    def get_type(self):
        return self.type


def post_process(prediction: torch.tensor):
    # identify specific zones in prediction:
    # pos_short: positive, len < 5, not at the end
    # neg_short: negative, len < 10, not at the start or end
    # pos_medium: positive, not short, len < 55
    # pos_long: positive, 55 <= len <= 240
    # pos_valid: positive, len > 240
    # neg_valid: negative, len >= 10
    zones = []
    start, value = 0, prediction[0]
    for i, residue in enumerate(prediction):
        if residue != value:  # save new zone
            zones.append(Zone(start=start, end=i, value=value, last=False))
            start = i
            value = residue
    zones.append(Zone(start=start, end=len(prediction), value=value, last=True))

    # change prediction according to rules:
    # 1. pos_short is changed to (-->) 0s   (already done during zone creation)
    # 2. pos_medium, neg_short, pos_medium --> 0s, (0s), 0s
    # 3. pos_medium, neg_short, pos_long/pos_valid --> 0s, (0s), (1s)
    # 4. pos_long/pos_valid, neg_short, pos_medium --> (1s), (0s), 0s
    # 5. pos_long/pos_valid, neg_short, pos_long/pos_valid --> (1s), 1s, (1s)
    # 6. pos_short, neg_short, pos_medium/pos_long --> (0s), (0s), 0s
    # 7. pos_medium/pos_long, neg_short, pos_short --> 0s, (0s), (0s)
    for i, zone in enumerate(zones):
        try:
            if zone.get_type() == "pos_medium" and zones[i + 1].get_type() == "neg_short":
                # case 2
                if zones[i + 2].get_type() == "pos_medium":
                    zone.set_value(0.0)
                    zones[i + 2].set_value(0.0)
                # cases 3 or (7)
                elif zones[i + 2].get_type() == "pos_long" or zones[i + 2].get_type() == "pos_valid" or \
                        zones[i + 2].get_type() == "pos_short":
                    zone.set_value(0.0)
            elif (zone.get_type() == "pos_long" or zone.get_type() == "pos_valid") and zones[
                i + 1].get_type() == "neg_short":
                # case 4
                if zones[i + 2].get_type() == "pos_medium":
                    zones[i + 2].set_value(0.0)
                # case 5
                elif zones[i + 2].get_type() == "pos_long" or zones[i + 2].get_type() == "pos_valid":
                    zones[i + 1].set_value(1.0)
                # case (7)
                elif zones[i + 2].get_type() == "pos_short":
                    zone.set_value(0.0)
            # case 6
            elif zone.get_type() == "pos_short" and zones[i + 1].get_type() == "neg_short" and \
                    (zones[i + 2].get_type() == "pos_medium" or zones[i + 2].get_type() == "pos_long"):
                zone.set_value(0.0)

        except IndexError:
            pass

    # put new prediction together
    prediction_pp = np.empty(0)
    for zone in zones:
        prediction_pp = np.append(prediction_pp, np.repeat(zone.get_value(), zone.get_length()))
    return torch.tensor(prediction_pp)


def predictFNN(embeddings, dataset_dir, cutoff, fold, mode, multilabel, post_processing, test, model_name, batch_size,
               dropout):
    output_name = f"../results/logs/predict_val_{model_name}_{fold}_{cutoff}.txt" if not test else \
        f"../results/logs/predict_val_{model_name}_{cutoff}_test.txt"

    with open(output_name, "w") as output_file:
        if not test:
            print("Fold: " + str(fold))
            # for validation use the training IDs in the current fold
            # read target data y and disorder information
            # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
            val_labels = read_labels(fold, None, dataset_dir)
        else:
            val_labels = read_labels(None, None, dataset_dir)

        # create the input and target data exactly how it's fed into the ML model
        # and add the confounding feature of disorder to the embeddings
        this_fold_val_input, this_fold_val_target, disorder_labels = get_ML_data(val_labels, embeddings, 'FNN', mode, multilabel, None)

        # instantiate the dataset
        validation_dataset = BindingDataset(this_fold_val_input, this_fold_val_target, disorder_labels, 'FNN')

        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_size = 1025 if mode == 'all' or multilabel else 1024
        output_size = 3 if multilabel else 1
        model = FNN(input_size, output_size, dropout, multilabel).to(device)
        model.load_state_dict(
            torch.load(f"../results/models/binding_regions_model_{model_name}_fold_{fold}.pth"))
        # test performance again, should be the same

        test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        all_prediction_act = list()
        all_prediction_max = list()
        all_labels = list()
        with torch.no_grad():
            for i, (input, label, _) in enumerate(test_loader):
                if multilabel:
                    input, label = input.to(device), label.to(device).T
                    label = [transform_output(
                        int(label[0][i]), int(label[1][i]), int(label[2][i]))
                        for i, _ in enumerate(label[0])]
                    all_labels.extend(label)
                    prediction = model(input).T
                    # apply activation function to prediction to enable classification
                    prediction_max_p = prediction[0] > cutoff[0]
                    prediction_max_n = prediction[1] > cutoff[1]
                    prediction_max_o = prediction[2] > cutoff[2]
                    prediction_max = [transform_output(
                        int(prediction_max_p[i]), int(prediction_max_n[i]), int(prediction_max_o[i]))
                        for i, _ in enumerate(prediction_max_p)]
                    all_prediction_max.extend(prediction_max)
                else:
                    input, label = input.to(device), label[:, None].to(device)
                    all_labels.extend(label.flatten().tolist())
                    prediction = model(input)
                    prediction_act = torch.sigmoid(prediction)
                    all_prediction_act.extend(prediction_act.flatten().tolist())
                    # apply activation function to prediction to enable classification
                    prediction_max = prediction_act > cutoff
                    all_prediction_max.extend(prediction_max.flatten().tolist())

        # group residues back to proteins again
        delimiter_0 = 0
        delimiter_1 = 0
        for p_id in val_labels.keys():
            if mode == 'disorder_only':
                delimiter_1 += len(str(val_labels[p_id][2]).replace('-', ''))
            elif mode == 'all':
                delimiter_1 += len(val_labels[p_id][2])
            if multilabel:
                output_file.write(f'{p_id}\nlabels:\t{"".join(all_labels[delimiter_0: delimiter_1])}'
                                  f'\nprediction:\t{"".join(all_prediction_max[delimiter_0: delimiter_1])}\n')
            else:
                output_file.write(f'{p_id}\nlabels:\t{torch.tensor(all_labels[delimiter_0: delimiter_1])}')
                # f'\nprediction_0:\t{torch.tensor(all_prediction_act[delimiter_0 : delimiter_1])}'
                # print(f'{p_id}\nprediction_0:\t{torch.tensor(all_prediction_act[delimiter_0: delimiter_1])}')
                output_file.write(f'\nprediction_1:\t{torch.tensor(all_prediction_max[delimiter_0: delimiter_1])}')
                if post_processing:
                    output_file.write(
                        f'\nprediction_pp:\t{post_process(torch.tensor(all_prediction_max[delimiter_0: delimiter_1]))}\n')

            delimiter_0 = delimiter_1


def investigate_cutoffs(train_embeddings: str, dataset_dir: str, model_name: str, mode: str = 'all', n_splits: int = 5,
                        architecture: str = 'FNN', n_layers: int = 0, kernel_size: int =5, batch_size: int = 512,
                        cutoff_percent_min: int = 0,
                        cutoff_percent_max: int = 100, step_percent: int = 5, multilabel: bool = False,
                        dropout: float = 0.3):
    """
    cutoffs for a classification of the predictions are tried out, careful: expensive!
    :param train_embeddings: path to the embedding file of the train set datapoints
    :param dropout: dropout probability
    :param model_name: name of the model
    :param mode: residues considered in the model, 'all' or 'disorder_only'
    :param n_splits: number of splits in cross validation
    :param architecture: 'CNN' or 'FNN'
    :param n_layers: number of layers in the CNN, must be 2, 5 or 8, is ignored if architecture == 'FNN'
    :param kernel_size: number of neighboring datapoints considered for prediction
    :param batch_size: batch_size, ignored if architecture == 'CNN'
    :param cutoff_percent_min: int between 0 and 100, this value / 100 = minimal cutoff that is tried out
    :param cutoff_percent_max: int between 0 and 100, this value / 100 = maximal cutoff that is tried out
    :param step_percent: int between 1 and 100, this value / 100 = step size for the cutoffs between cutoff_percent_min
    and cutoff_percent_max
    :param multilabel: True if it is a multilabel predictor, else False
    """
    # read input embeddings
    embeddings = dict()
    with h5py.File(train_embeddings, 'r') as f:
        for key, embedding in f.items():
            original_id = embedding.attrs['original_id']
            embeddings[original_id] = np.array(embedding)
    # now {IDs: embeddings} are written in the embeddings dictionary

    try_cutoffs(model_name, dataset_dir, embeddings, mode, multilabel, n_splits, architecture, n_layers, kernel_size,
                batch_size, cutoff_percent_min, cutoff_percent_max, step_percent, dropout)


def predict(train_embeddings: str, dataset_dir: str, test_embeddings: str, model_name: str, fold: int, cutoff,
            mode: str = 'all',
            architecture: str = 'FNN', n_layers: int = 0, kernel_size: int = 5, batch_size: int = 512,
            multilabel: bool = False, dropout: float = 0.3, test: bool = False, post_processing: bool = True):
    """
    make a final prediction on a validation (set fold!) or test set using the newly determined cutoff
    :param dataset_dir: directory where the dataset files are stored
    :param train_embeddings: path to the embedding file of the train set datapoints
    :param test_embeddings: path to the embedding file of the test set datapoints
    :param post_processing: do the optional the post-processing step?
    :param test: prediction on the test set?
    :param cutoff: single float or list of floats[p, n, o] for classification cutoffs, between 0.0 and 1.0
    :param fold: fold ID of the validation fold, ignored if test==True
    :param model_name: name of the used model
    :param mode: residues considered in the model, 'all' or 'disorder_only'
    :param architecture: ML architecture, 'CNN' or 'FNN'
    :param n_layers: number of convolutional layers used in the CNN, ignored if architecture=='FNN'
    :param kernel_size: number of neighboring datapoints considered for prediction
    :param batch_size: batch_size
    :param multilabel: multilabel classifier?
    :param dropout: float between 0.0 and 1.0, dropout chance in the used model.
    """

    # read input embeddings
    embeddings_in = test_embeddings if test else train_embeddings
    embeddings = dict()
    with h5py.File(embeddings_in, 'r') as f:
        for key, embedding in f.items():
            original_id = embedding.attrs['original_id']
            embeddings[original_id] = np.array(embedding)
    # now {IDs: embeddings} are written in the embeddings dictionary

    # get predictions for chosen cutoff, fold
    if architecture == 'CNN':
        predictCNN(embeddings, dataset_dir, cutoff, fold, model_name, n_layers, kernel_size, dropout, mode, test)
    elif architecture == 'FNN':
        predictFNN(embeddings, dataset_dir, cutoff, fold, mode, multilabel, post_processing, test, model_name,
                   batch_size, dropout)
    else:
        raise ValueError("architecture must be 'CNN' or 'FNN'")

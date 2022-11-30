"""
comparable performance assessment of all models (except 3_disorder_only, is not comparable):
bootstrapping, avg over all folds, all relevant metrics
not part of pipeline
"""

import numpy as np
import h5py
from Bio import SeqIO
import re
import torch.tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import nn
from scipy import stats
from pathlib import Path
from os.path import exists


def read_labels(fold, oversampling, dataset_dir):
    if fold is None:   # --> test set
        file_name = f'../dataset/test_set_input.txt'
    else:
        if oversampling is None:  # no oversampling on validation set! (or mode with no oversampling)
            file_name = f'{dataset_dir}folds/CV_fold_{fold}_labels.txt'
            if not exists(file_name):
                file_name = f'{dataset_dir}folds/CV_fold_{fold}_labels_None.txt'
        else:
            file_name = f'{dataset_dir}folds/CV_fold_{fold}_labels_{oversampling}.txt'
    with open(file_name) as handle:
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


def get_ML_data(labels, embeddings, architecture, mode, multilabel):
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
            emb_with_conf = np.column_stack((embeddings[id], conf_feature))
            input.append(emb_with_conf)
        elif mode == 'disorder_only':
            if architecture == 'FNN':
                bool_list = [False if x == '-' else True for x in list(labels[id][2])]
                input.append(embeddings[id][bool_list])
                disorder.append([1] * len(labels[id][2]))
            else:  # CNN
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
                        disorder.append([1] * (i - diso_start))
                        # disorder.append(list('1' * (i-diso_start)))
                        starts.append(diso_start)
                        stops.append(i)
                        # print(f'{id}: region from {diso_start} to {i}')
                    current = diso_residue
                # disorder at final residue?
                if current:
                    input.append(embeddings[id][diso_start:])
                    disorder.append([1] * (len(bool_list) - diso_start))
                    # disorder.append(list('1' * (len(bool_list)-diso_start)))
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
        padding = int((kernel_size - 1) / 2)
        self.n_layers = n_layers
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


def criterion(loss_func, prediction, label):  # sum over all classification heads
    losses = 0
    prediction = prediction.T
    label = label.T
    for i, _ in enumerate(prediction):  # for each class (-> 1-dimensional loss)
        losses += loss_func(prediction[i], label[i])
    return losses


def conf_matrix(prediction, label, disorder, batch_wise_loss, batch_wise, fold, random):
    if fold is not None:
        batch_wise_loss.append(loss_function(prediction, label.to(torch.float32)).item())
    # apply activation function to prediction to enable classification and transpose matrices
    if random:
        prediction_act = prediction     # no sigmoid needed for these values
    else:
        prediction_act = torch.sigmoid(prediction)
    if fold is not None:
        prediction_max = prediction_act > cutoff[fold]
    else:
        prediction_max = prediction

    # confusion matrix values
    batch_wise["correct"].append((prediction_max == label).type(torch.float).sum().item())
    batch_wise["TP"].append(
        (prediction_max == label)[label == 1].type(torch.float).sum().item())
    batch_wise["FP"].append(
        (prediction_max != label)[label == 0].type(torch.float).sum().item())
    batch_wise["TN"].append(
        (prediction_max == label)[label == 0].type(torch.float).sum().item())
    batch_wise["FN"].append(
        (prediction_max != label)[label == 1].type(torch.float).sum().item())

    # metrics within disorder
    batch_wise["diso_correct"].append((prediction_max == label)[disorder == 1].type(torch.float).sum().item())
    mask_0 = (label == 0) & (disorder == 1)
    mask_1 = (label == 1) & (disorder == 1)
    batch_wise["diso_TP"].append((prediction_max == label)[mask_1].type(torch.float).sum().item())
    batch_wise["diso_FP"].append((prediction_max != label)[mask_0].type(torch.float).sum().item())
    batch_wise["diso_TN"].append((prediction_max == label)[mask_0].type(torch.float).sum().item())
    batch_wise["diso_FN"].append((prediction_max != label)[mask_1].type(torch.float).sum().item())

    return batch_wise_loss, batch_wise


def metrics(confusion_dict):
    # precision, recall, neg.precision, neg.recall, F1, MCC, balanced acc
    precision = confusion_dict["TP"] / (confusion_dict["TP"] + confusion_dict["FP"])
    neg_precision = confusion_dict["TN"] / (confusion_dict["TN"] + confusion_dict["FN"])
    recall = confusion_dict["TP"] / (confusion_dict["TP"] + confusion_dict["FN"])
    neg_recall = confusion_dict["TN"] / (confusion_dict["TN"] + confusion_dict["FP"])
    specificity = confusion_dict["TN"] / (confusion_dict["TN"] + confusion_dict["FP"])
    balanced_acc = (recall + specificity) / 2
    f1 = 2 * ((precision * recall) / (precision + recall))
    mcc = (confusion_dict["TN"] * confusion_dict["TP"] - confusion_dict["FP"] * confusion_dict["FN"]) / \
          np.sqrt((confusion_dict["TN"] + confusion_dict["FN"]) * (confusion_dict["FP"] + confusion_dict["TP"]) *
                  (confusion_dict["TN"] + confusion_dict["FP"]) * (confusion_dict["FN"] + confusion_dict["TP"]))
    # analog in disorder
    precision_D = confusion_dict["diso_TP"] / (confusion_dict["diso_TP"] + confusion_dict["diso_FP"])
    neg_precision_D = confusion_dict["diso_TN"] / (confusion_dict["diso_TN"] + confusion_dict["diso_FN"])
    recall_D = confusion_dict["diso_TP"] / (confusion_dict["diso_TP"] + confusion_dict["diso_FN"])
    neg_recall_D = confusion_dict["diso_TN"] / (confusion_dict["diso_TN"] + confusion_dict["diso_FP"])
    specificity_D = confusion_dict["diso_TN"] / (confusion_dict["diso_TN"] + confusion_dict["diso_FP"])
    balanced_acc_D = (recall_D + specificity_D) / 2
    f1_D = 2 * ((precision_D * recall_D) / (precision_D + recall_D))
    mcc_D = (confusion_dict["diso_TN"] * confusion_dict["diso_TP"] - confusion_dict["diso_FP"] * confusion_dict["diso_FN"]) / \
          np.sqrt((confusion_dict["diso_TN"] + confusion_dict["diso_FN"]) * (confusion_dict["diso_FP"] + confusion_dict["diso_TP"]) *
                  (confusion_dict["diso_TN"] + confusion_dict["diso_FP"]) * (confusion_dict["diso_FN"] + confusion_dict["diso_TP"]))
    return {"Precision": precision, "Recall": recall, "Neg_Precision": neg_precision, "Neg_Recall": neg_recall,
            "Balanced Acc.": balanced_acc, "F1": f1, "MCC": mcc,
            "D Precision": precision_D, "D Recall": recall_D, "D Neg_Precision": neg_precision_D, "D Neg_Recall": neg_recall_D,
            "D Balanced Acc.": balanced_acc_D, "D F1": f1_D, "D MCC": mcc_D}


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
    prediction = prediction.cpu()
    zones = []
    start, value = 0, prediction[0]
    for i, residue in enumerate(prediction):
        if residue != value:    # save new zone
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
            if zone.get_type() == "pos_medium" and zones[i+1].get_type() == "neg_short":
                # case 2
                if zones[i+2].get_type() == "pos_medium":
                    zone.set_value(0.0)
                    zones[i+2].set_value(0.0)
                # cases 3 or (7)
                elif zones[i+2].get_type() == "pos_long" or zones[i+2].get_type() == "pos_valid" or \
                        zones[i+2].get_type() == "pos_short":
                    zone.set_value(0.0)
            elif (zone.get_type() == "pos_long" or zone.get_type() == "pos_valid") and zones[i+1].get_type() == "neg_short":
                # case 4
                if zones[i+2].get_type() == "pos_medium":
                    zones[i+2].set_value(0.0)
                # case 5
                elif zones[i+2].get_type() == "pos_long" or zones[i+2].get_type() == "pos_valid":
                    zones[i+1].set_value(1.0)
                # case (7)
                elif zones[i+2].get_type() == "pos_short":
                    zone.set_value(0.0)
            # case 6
            elif zone.get_type() == "pos_short" and zones[i+1].get_type() == "neg_short" and \
                    (zones[i+2].get_type() == "pos_medium" or zones[i+2].get_type() == "pos_long"):
                zone.set_value(0.0)

        except IndexError:
            pass

    # put new prediction together
    prediction_pp = np.empty(0)
    for zone in zones:
        prediction_pp = np.append(prediction_pp, np.repeat(zone.get_value(), zone.get_length()))
    return torch.tensor(prediction_pp)[:, None]


def assess(name, cutoff, mode, multilabel, architecture, n_layers, kernel_size, batch_size, loss_function, post_processing, test, best_fold, dataset_dir):
    # predict and assess performance of 1 model
    if multilabel:
        all_conf_matrices = [{"correct": [], "TP": [], "FP": [], "TN": [], "FN": []},
                             {"correct": [], "TP": [], "FP": [], "TN": [], "FN": []},
                             {"correct": [], "TP": [], "FP": [], "TN": [], "FN": []}]
    else:
        all_conf_matrices = {"correct": [], "TP": [], "FP": [], "TN": [], "FN": [],
                             "diso_correct": [], "diso_TP": [], "diso_FP": [], "diso_TN": [], "diso_FN": []}
    if test:
        cutoff = [cutoff[best_fold]]
    for fold, _ in enumerate(cutoff):
        if test:
            print(f"{name} Fold {best_fold}")
            # read target data y and disorder information
            # re-format input information to 3 sequences in a list per protein in dict val_labels{}
            val_labels = read_labels(None, None, dataset_dir)
        else:
            print(f"{name} Fold {fold}")
            # for validation use the training IDs in the current fold
            # read target data y and disorder information
            # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
            val_labels = read_labels(fold, None, dataset_dir)

        # create the input and target data exactly how it's fed into the ML model
        # and add the confounding feature of disorder to the embeddings
        this_fold_val_input, this_fold_val_target, this_fold_disorder = \
            get_ML_data(val_labels, embeddings, architecture, mode, multilabel)

        # instantiate the dataset
        validation_dataset = BindingDataset(this_fold_val_input, this_fold_val_target, this_fold_disorder, architecture)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_size = 1025 if mode == 'all' else 1024
        output_size = 3 if multilabel else 1

        if architecture == "FNN":
            model = FNN(input_size, output_size, dropout, multilabel).to(device)
        else:
            model = CNN(n_layers, kernel_size, dropout, input_size).to(device)

        if name.startswith("random"):
            model = None
        else:
            model.load_state_dict(
                torch.load(f"../results/models/binding_regions_model_{name}_fold_{fold}.pth"))

        if architecture == "CNN":
            batch_size = 1
        test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        if model is not None:
            model.eval()
        with torch.no_grad():
            if multilabel:
                # save confusion matrix values for each batch
                batch_wise_loss = []
                batch_wise_p = {"correct": [], "TP": [], "FP": [], "TN": [], "FN": []}
                batch_wise_n = {"correct": [], "TP": [], "FP": [], "TN": [], "FN": []}
                batch_wise_o = {"correct": [], "TP": [], "FP": [], "TN": [], "FN": []}

                for input, label in test_loader:
                    input, label = input.to(device), label.to(device)
                    if name.startswith("random"):
                        prediction = torch.rand(label.T.shape).to(device)
                    else:
                        prediction = model(input, multilabel).T
                    label = label.T
                    batch_wise_loss.append(criterion(loss_function, prediction, label.to(torch.float32)))
                    # apply activation function to prediction to enable classification and transpose matrices
                    prediction_max_p = prediction[0] > cutoff[fold][0]
                    prediction_max_n = prediction[1] > cutoff[fold][1]
                    prediction_max_o = prediction[2] > cutoff[fold][2]
                    prediction_max = [prediction_max_p, prediction_max_n, prediction_max_o]

                    # confusion matrix values
                    batch_wise_p["correct"].append((prediction_max[0] == label[0]).type(torch.float).sum().item())
                    batch_wise_p["TP"].append(
                        (prediction_max[0] == label[0])[label[0] == 1].type(torch.float).sum().item())
                    batch_wise_p["FP"].append(
                        (prediction_max[0] != label[0])[label[0] == 0].type(torch.float).sum().item())
                    batch_wise_p["TN"].append(
                        (prediction_max[0] == label[0])[label[0] == 0].type(torch.float).sum().item())
                    batch_wise_p["FN"].append(
                        (prediction_max[0] != label[0])[label[0] == 1].type(torch.float).sum().item())

                    batch_wise_n["correct"].append((prediction_max[1] == label[1]).type(torch.float).sum().item())
                    batch_wise_n["TP"].append(
                        (prediction_max[1] == label[1])[label[1] == 1].type(torch.float).sum().item())
                    batch_wise_n["FP"].append(
                        (prediction_max[1] != label[1])[label[1] == 0].type(torch.float).sum().item())
                    batch_wise_n["TN"].append(
                        (prediction_max[1] == label[1])[label[1] == 0].type(torch.float).sum().item())
                    batch_wise_n["FN"].append(
                        (prediction_max[1] != label[1])[label[1] == 1].type(torch.float).sum().item())

                    batch_wise_o["correct"].append((prediction_max[2] == label[2]).type(torch.float).sum().item())
                    batch_wise_o["TP"].append(
                        (prediction_max[2] == label[2])[label[2] == 1].type(torch.float).sum().item())
                    batch_wise_o["FP"].append(
                        (prediction_max[2] != label[2])[label[2] == 0].type(torch.float).sum().item())
                    batch_wise_o["TN"].append(
                        (prediction_max[2] == label[2])[label[2] == 0].type(torch.float).sum().item())
                    batch_wise_o["FN"].append(
                        (prediction_max[2] != label[2])[label[2] == 1].type(torch.float).sum().item())

                for k in batch_wise_p.keys():
                    all_conf_matrices[0][k] = np.append(all_conf_matrices[0][k], batch_wise_p[k])
                    all_conf_matrices[1][k] = np.append(all_conf_matrices[1][k], batch_wise_n[k])
                    all_conf_matrices[2][k] = np.append(all_conf_matrices[2][k], batch_wise_o[k])


            else:  # not multilabel
                # save confusion matrix values for each protein
                batch_wise_loss = []
                batch_wise = {"correct": [], "TP": [], "FP": [], "TN": [], "FN": [],
                              "diso_correct": [], "diso_TP": [], "diso_FP": [], "diso_TN": [], "diso_FN": []}

                for input, label, disorder in test_loader:
                    input, label, disorder = input.to(device), label[:, None].to(device), disorder[:, None].to(device)
                    if name.startswith("random"):
                        prediction = torch.rand(len(label), 1).to(device)
                        random = True
                    else:
                        prediction = model(input)
                        random = False
                        if post_processing:
                            prediction = post_process(prediction).to(device)
                    batch_wise_loss, batch_wise = conf_matrix(prediction, label, disorder, batch_wise_loss, batch_wise, fold, random)

                for k in batch_wise.keys():
                    all_conf_matrices[k] = np.append(all_conf_matrices[k], batch_wise[k])

    # metrics and sd over all folds/proteins
    if multilabel:
        # batch-wise metrics
        all_metrics = [metrics(all_conf_matrices[0]), metrics(all_conf_matrices[1]), metrics(all_conf_matrices[2])]

        # exclude nan values
        for k in all_metrics[0].keys():
            all_metrics[0][k] = all_metrics[0][k][np.logical_not(np.isnan(all_metrics[0][k]))]
            all_metrics[1][k] = all_metrics[1][k][np.logical_not(np.isnan(all_metrics[1][k]))]
            all_metrics[2][k] = all_metrics[2][k][np.logical_not(np.isnan(all_metrics[2][k]))]

        # standard error calculation
        all_sd_errors = [{}, {}, {}]
        for k in all_metrics[0].keys():
            all_sd_errors[0][k] = np.std(all_metrics[0][k], ddof=1) / np.sqrt(len(all_metrics[0][k]))
            all_sd_errors[1][k] = np.std(all_metrics[1][k], ddof=1) / np.sqrt(len(all_metrics[1][k]))
            all_sd_errors[2][k] = np.std(all_metrics[2][k], ddof=1) / np.sqrt(len(all_metrics[2][k]))

        # calculate sum and absolute (avg) metrics
        sum_matrix = [{}, {}, {}]
        for k in all_conf_matrices[0].keys():
            sum_matrix[0][k] = np.sum(all_conf_matrices[0][k])
            sum_matrix[1][k] = np.sum(all_conf_matrices[1][k])
            sum_matrix[2][k] = np.sum(all_conf_matrices[2][k])

        avg_metrics = [metrics(sum_matrix[0]), metrics(sum_matrix[1]), metrics(sum_matrix[2])]


    else:  # no multilabel
        # protein-wise metrics
        all_metrics = metrics(all_conf_matrices)

        # exclude nan values
        for k in all_metrics.keys():
            all_metrics[k] = all_metrics[k][np.logical_not(np.isnan(all_metrics[k]))]

        # standard error calculation
        all_sd_errors = {}
        for k in all_metrics.keys():
            all_sd_errors[k] = np.std(all_metrics[k], ddof=1) / np.sqrt(len(all_metrics[k]))

        # calculate sum and absolute (avg) metrics
        sum_matrix = {}
        for k in all_conf_matrices.keys():
            sum_matrix[k] = np.sum(all_conf_matrices[k])

        avg_metrics = metrics(sum_matrix)


    return sum_matrix, avg_metrics, all_sd_errors, all_metrics


def assess_bindEmbed():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_folder = "../results/bindEmbed21DL_predictions"
    predictions_binary = {}
    for p in Path(in_folder).glob('*.bindPredict_out'):
        with p.open() as f:
            long_table = f.readlines()[1:]
            prediction = torch.tensor([0 if row.split("\t")[7] == 'nb\n' else 1 for row in long_table]).to(device)
            predictions_binary[p.name[:-16]] = prediction

    all_conf_matrices = {"correct": [], "TP": [], "FP": [], "TN": [], "FN": []}
    val_labels = read_labels(None, None, dataset_dir)

    target = []
    predictions = []
    for id in predictions_binary.keys():
        # modify bindEMbed predictions to predict in disordered regions only
        pr = predictions_binary[id]
        try:
            disorder = (val_labels[id][1])
            indices_to_change = [i if d != "D" else None for i, d in enumerate(disorder)]
            for i in indices_to_change:
                if i is not None:
                    pr[i] = 0
            predictions.append(pr)

            # for target: 0 = non-binding, 1 = binding, 0 = not in disordered region
            binding = str(val_labels[id][2])
            binding = re.sub(r'-|_', '0', binding)
            binding = list(re.sub(r'P|N|O|X|Y|Z|A', '1', binding))
            binding = np.array(binding, dtype=float)
            target.append(binding)
        except KeyError:
            print(f"Warning: {id} not found in val_labels")

    # save confusion matrix values for each protein
    batch_wise_loss = []
    batch_wise = {"correct": [], "TP": [], "FP": [], "TN": [], "FN": []}

    for i, label in enumerate(target):
        pr = predictions[i].to(device)
        label = torch.tensor(label).to(device)
        batch_wise_loss, batch_wise = conf_matrix(pr, label, disorder, batch_wise_loss, batch_wise, None, False)

    for k in batch_wise.keys():
        all_conf_matrices[k] = np.append(all_conf_matrices[k], batch_wise[k])

    # metrics and sd over all folds/proteins
    # protein-wise metrics
    all_metrics = metrics(all_conf_matrices)

    # exclude nan values
    for k in all_metrics.keys():
        all_metrics[k] = all_metrics[k][np.logical_not(np.isnan(all_metrics[k]))]

    # standard error calculation
    all_sd_errors = {}
    for k in all_metrics.keys():
        all_sd_errors[k] = np.std(all_metrics[k], ddof=1) / np.sqrt(len(all_metrics[k]))

    # calculate sum and absolute (avg) metrics
    sum_matrix = {}
    for k in all_conf_matrices.keys():
        sum_matrix[k] = np.sum(all_conf_matrices[k])

    avg_metrics = metrics(sum_matrix)

    return sum_matrix, avg_metrics, all_sd_errors



if __name__ == '__main__':
    # read input embeddings
    test = False
    embeddings_in = '../dataset/MobiDB_dataset/test_set.h5' if test else '../dataset/MobiDB_dataset/train_set.h5'
    embeddings = dict()
    with h5py.File(embeddings_in, 'r') as f:
        for key, embedding in f.items():
            original_id = embedding.attrs['original_id']
            embeddings[original_id] = np.array(embedding)
    # now {IDs: embeddings} are written in the embeddings dictionary

    """ disprot code
    variants = [0.0, 1.0, 2.0, 2.1, 2.20, 2.21, 2.213, 12.0, 3.0, 13.0, 4.0, 4.1, 14.0]
    # cutoffs are different for each fold, variant (and class, if multiclass)!
    cutoffs = {0.0: [0.315, 0.16, 0.235, 0.245, 0.39],
               1.0: [0.005, 0.005, 0.005, 0.005, 0.01],
               2.0: [0.12, 0.1, 0.06, 0.06, 0.08],
               2.1: [0.05, 0.05, 0.15, 0.2, 0.85],
               2.20: [0.65, 0.65, 0.65, 0.5, 0.8],
               2.21: [0.8, 0.8, 0.85, 0.85, 0.85],
               2.211: [0.7, 0.6, 0.65, 0.55, 0.6],
               2.212: [0.8, 0.85, 0.75, 0.8, 0.85],
               2.213: [0.8, 0.8, 0.85, 0.85, 0.85],
               12.0: [0.902, 0.902, 0.902, 0.902, 0.902],  # here cutoff = chance of negative prediction
               3.0: [0.44, 0.4, 0.48, 0.48, 0.5],
               13.0: [0.578, 0.578, 0.578, 0.578, 0.578],  # here cutoff = chance of negative prediction
               4.0: [[0.6, 0.15, 0.05], [0.6, 0.15, 0.1], [0.6, 0.15, 0.1], [0.15, 0.15, 0.15], [0.65, 0.1, 0.1]],
               4.1: [[0.3, 0.4, 0.25], [0.3, 0.4, 0.25], [0.3, 0.45, 0.25], [0.3, 0.45, 0.2], [0.5, 0.1, 0.4]],
               14.0: [[0.923, 0.983, 0.986], [0.923, 0.983, 0.986], [0.923, 0.983, 0.986], [0.923, 0.983, 0.986],
                      [0.923, 0.983, 0.986]]}  # here cutoff = chance of negative prediction
    names = {0.0: "0_simple_without_dropout",
             1.0: "1_5layers",
             2.0: "2_FNN",
             2.1: "2-1_new_oversampling",
             2.20: "2-2_dropout_0.2_new",
             2.21: "2-2_dropout_0.3_new",
             2.211: "2-2_dropout_0.3_lr_0.005",
             2.212: "2-2_dropout_0.3_lr_0.008",
             2.213: "2-2_dropout_0.3_new",
             12.0: "random_binary",
             3.0: "3_d_only",
             13.0: "random_d_only",
             4.0: "4_multiclass",
             4.1: "4-1_new_oversampling",
             14.0: "random_multilabel"}

    best_folds = {0.0: 0,
                  1.0: 4,
                  2.0: 0,
                  2.1: 0,
                  2.20: 0,
                  2.21: 4,
                  2.211: 0,
                  2.212: 4,
                  2.213: 4,
                  12.0: 0,
                  3.0: 4,
                  13.0: 0,
                  4.0: 0,
                  4.1: 2,
                  14.0: 0}

    """

    # mobidb code
    dataset_dir = '../dataset/MobiDB_dataset/'
    # variants = [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.0005, 2.001, 2.02, 2.03, 2.06, 2.07, 2.08, 2.1, 2.2, 3.0, 3.1, 3.2, 3.3, 3.4, 10.0, 12.0]
    variants = [2.0, 2.06, 2.07]
    assessment_name = "mobidb_D_CNN_0_kernel_size"      # "mobidb" / "2.21_only" / ""
    # cutoffs are different for each fold and variant
    cutoffs = {0.0: [0.35, 0.3, 0.3, 0.15, 0.4],
               0.1: [0.2, 0.2, 0.3, 0.3, 0.4],
               0.2: [0.1, 0.15, 0.25, 0.15, 0.05],
               1.0: [0.4, 0.25, 0.4, 0.3, 0.45],
               1.1: [0.9, 0.9, 0.9, 0.9, 0.85],
               1.2: [0.55, 0.55, 0.6, 0.6, 0.65],
               1.3: [0.4, 0.5, 0.45, 0.2, 0.4],
               1.4: [0.25, 0.3, 0.3, 0.35, 0.25],
               1.5: [0.65, 0.55, 0.45, 0.5, 0.55],
               2.0: [0.4, 0.35, 0.45, 0.35, 0.35],
               2.0005: [0.45, 0.45, 0.5, 0.45, 0.2],
               2.001: [0.5, 0.5, 0.5, 0.55, 0.25],
               2.02: [0.55, 0.55, 0.55, 0.55, 0.5],
               2.03: [0.15, 0.5, 0.25, 0.55, 0.3],
               2.06: [0.45, 0.4, 0.5, 0.5, 0.3],
               2.07: [0.2, 0.2, 0.45, 0.35, 0.5],
               2.08: [0.25, 0.25, 0.25, 0.5, 0.25],
               2.1: [0.2, 0.4, 0.45, 0.55, 0.2],
               2.2: [0.5, 0.6, 0.5, 0.45, 0.35],
               3.0: [0.4, 0.45, 0.45, 0.45, 0.4],
               3.1: [0.5, 0.5, 0.55, 0.5, 0.5],
               3.2: [0.4, 0.4, 0.4, 0.45, 0.4],
               3.3: [0.5, 0.5, 0.45, 0.5, 0.45],
               3.4: [0.6, 0.55, 0.55, 0.5, 0.5],
               10.0: [0.94, 0.94, 0.94, 0.94, 0.94],
               12.0: [0.63, 0.63, 0.63, 0.63, 0.63]
               }
    names = {0.0: "mobidb_CNN_0",
             0.1: "mobidb_CNN_1",
             0.2: "mobidb_CNN_2",
             1.0: "mobidb_FNN_0",
             1.1: "mobidb_FNN_1",
             1.2: "mobidb_FNN_2",
             1.3: "mobidb_FNN_3",
             1.4: "mobidb_FNN_4",
             1.5: "mobidb_FNN_5",
             2.0: "mobidb_D_CNN_0",
             2.0005: "mobidb_D_CNN_0_lr0005",
             2.001: "mobidb_D_CNN_0_lr001",
             2.02: "mobidb_D_CNN_0_d2",
             2.03: "mobidb_D_CNN_0_d3",
             2.06: "mobidb_D_CNN_0_k3",
             2.07: "mobidb_D_CNN_0_k7",
             2.08: "mobidb_D_CNN_0_l8",
             2.1: "mobidb_D_CNN_1",
             2.2: "mobidb_D_CNN_2",
             3.0: "mobidb_D_FNN_0",
             3.1: "mobidb_D_FNN_1",
             3.2: "mobidb_D_FNN_2",
             3.3: "mobidb_D_FNN_3",
             3.4: "mobidb_D_FNN_4",
             10.0: "random_binary",
             12.0: "random_d_only",
             }

    best_folds = {0.0: 2,
                  0.1: 2,
                  0.2: 0,
                  1.0: 2,
                  1.1: 3,
                  1.2: 2,
                  1.3: 2,
                  1.4: 0,
                  1.5: 2,
                  2.0: 2,
                  2.0005: 2,
                  2.001: 2,
                  2.02: 4,
                  2.03: 3,
                  2.06: 2,
                  2.07: 2,
                  2.08: 3,
                  2.1: 2,
                  2.2: 3,
                  3.0: 4,
                  3.1: 2,
                  3.2: 4,
                  3.3: 1,
                  3.4: 0,
                  10.0: 2,
                  12.0: 2,
                  }



    performances = []
    per_model_metrics = []
    for variant in variants:
        # set parameters
        print(f"variant {variant}:")

        """ disprot implementation
        mode = 'disorder_only' if (variant % 10) == 3.0 else 'all'
        multilabel = True if (variant % 10) >= 4.0 else False
        network = 'CNN' if (variant % 10) < 2.0 else 'FNN'
        if variant == 2.20:
            dropout = 0.2
        elif 2.21 <= variant < 3.0:
            dropout = 0.3
        else:
            dropout = 0.0
        if variant == 2.213:
            post_processing = True
        else:
            post_processing = False
        """
        # mobidb implementation
        mode = 'disorder_only' if (variant % 10) >= 2.0 else 'all'
        multilabel = False
        architecture = 'CNN' if 'CNN' in names[variant] else 'FNN'  # 0, 1, 5, 6
        if 'k7' in names[variant]:
            kernel_size = 7
        elif 'k3' in names[variant]:
            kernel_size = 3
        else:
            kernel_size = 5
        if 'l8' in names[variant]:
            n_layers = 8
        else:
            n_layers = 5
        dropout = 0.0
        post_processing = False
        batch_size = 512


        loss_function = nn.BCELoss() if multilabel else nn.BCEWithLogitsLoss()
        cutoff = cutoffs[variant]
        name = names[variant]
        best_fold = best_folds[variant]

        assessment = assess(name, cutoff, mode, multilabel, architecture, n_layers, kernel_size, batch_size,
                            loss_function, post_processing, test, best_fold, dataset_dir)
        performances.append(assessment[:-1])
        per_model_metrics.append(assessment[-1])


    # Welch test for some specific models
    """
    print("Welch test, with vs without post-processing")
    for k in per_model_metrics[0].keys():
        print(k, "(statistic, pvalue)")
        print(stats.ttest_ind(per_model_metrics[5][k], per_model_metrics[6][k], equal_var=True))
    """

    if test:
        bindEmbed_performance = assess_bindEmbed()

    output_name = f'../results/logs/performance_assessment_{assessment_name}.tsv' if not test else \
        f'../results/logs/performance_assessment_test_{assessment_name}.tsv'

    with open(output_name, "w") as output:
        output.write("model\tclass\t")
        for key in performances[0][0].keys():  # conf-matrix
            output.write(str(key) + "\t")
        for key in performances[0][1].keys():  # metrics
            output.write(str(key) + "\t")
        for key in performances[0][2].keys():  # SEs of metrics
            output.write("SE_" + str(key) + "\t")
        output.write("\n")

        for i, v in enumerate(variants):
            if 'MobiDB' not in dataset_dir and (v % 10) >= 4.0:  # multiclass
                for j, b_class in enumerate(["protein", "nuc", "other"]):
                    output.write(f"{names[v]}\t{b_class}\t")
                    for key in performances[0][0].keys():  # conf-matrix
                        output.write(str(performances[i][0][j][key]) + "\t")
                    for key in performances[0][1].keys():  # metrics
                        output.write(str(performances[i][1][j][key]) + "\t")
                    for key in performances[0][2].keys():  # SEs of metrics
                        output.write(str(performances[i][2][j][key]) + "\t")
                    output.write("\n")
            else:
                output.write(f"{names[v]}\t-\t")
                for key in performances[0][0].keys():  # conf-matrix
                    output.write(str(performances[i][0][key]) + "\t")
                for key in performances[0][1].keys():  # metrics
                    output.write(str(performances[i][1][key]) + "\t")
                for key in performances[0][2].keys():  # SEs of metrics
                    output.write(str(performances[i][2][key]) + "\t")
                output.write("\n")

        if test:
            output.write("bindEmbed21DL\t-\t")
            for key in performances[0][0].keys():  # conf-matrix
                output.write(str(bindEmbed_performance[0][key]) + "\t")
            for key in performances[0][1].keys():  # metrics
                output.write(str(bindEmbed_performance[1][key]) + "\t")
            for key in performances[0][2].keys():  # SEs of metrics
                output.write(str(bindEmbed_performance[2][key]) + "\t")
            output.write("\n")

"""
comparable performance assessment of all models (except 3_disorder_only, is not comparable):
bootstrapping, avg over all folds, all relevant metrics
"""

import numpy as np
import h5py
from Bio import SeqIO
import re
import torch.tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import nn, optim
import datetime
import copy


def read_labels(fold, oversampling):
    if oversampling is None:  # no oversampling on validation set!
        file_name = f'../dataset/folds/CV_fold_{fold}_labels.txt'
    else:
        file_name = f'../dataset/folds/CV_fold_{fold}_labels_{oversampling}.txt'
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


def get_ML_data(labels, embeddings, mode, multilabel, new_datapoints):
    input = list()
    target = list()
    datapoint_counter = 0
    for id in labels.keys():
        if mode == 'all' or multilabel:
            conf_feature = str(labels[id][1])
            conf_feature = list(conf_feature.replace('-', '0').replace('D', '1'))
            conf_feature = np.array(conf_feature, dtype=float)
            if '*' not in id:
                emb_with_conf = np.column_stack((embeddings[id], conf_feature))
            else:  # data points created by residue-oversampling
                # use pre-computed embedding
                emb_with_conf = new_datapoints[datapoint_counter]
                datapoint_counter += 1
                if emb_with_conf.shape[0] != len(labels[id][1]):  # sanity check
                    raise ValueError(f'Wrong match between label and embedding. Label of {id} has length '
                                     f'{len(labels[id][1])}, emb has shape {emb_with_conf.shape}')

            input.append(emb_with_conf)
        elif mode == 'disorder_only':
            bool_list = [False if x == '-' else True for x in list(labels[id][2])]
            input.append(embeddings[id][bool_list])

        if not multilabel:
            # for target: 0 = non-binding, 1 = binding, 0 = not in disordered region (2 doesnt work!, would be multi-class)
            binding = str(labels[id][2])
            if mode == 'all':
                binding = re.sub(r'-|_', '0', binding)
            elif mode == 'disorder_only':
                binding = binding.replace('-', '').replace('_', '0')
            binding = list(re.sub(r'P|N|O|X|Y|Z|A', '1', binding))
            binding = np.array(binding, dtype=float)
            target.append(binding)
        else:
            # for target: 0 = non-binding or not in disordered region, 1 = binding, 3-dimensions per residue
            binding = str(labels[id][2])
            binding_encoded = [[], [], []]
            binding_encoded[0] = list(re.sub(r'P|X|Y|A', '1', re.sub(r'-|_|N|O|Z', '0', binding)))  # protein-binding?
            binding_encoded[1] = list(
                re.sub(r'N|X|Z|A', '1', re.sub(r'-|_|P|O|Y', '0', binding)))  # nucleic-acid-binding?
            binding_encoded[2] = list(re.sub(r'O|Y|Z|A', '1', re.sub(r'-|_|P|N|X', '0', binding)))  # other-binding?
            target.append(np.array(binding_encoded, dtype=float).T)

        """
        if id == 'P17947*':
            print(input[-1])
            print(input[-1].shape)
            print(binding)
        """

    return input, target


# build the dataset
class BindingDataset(Dataset):
    def __init__(self, embeddings, binding_labels, network):
        self.inputs = embeddings
        self.labels = binding_labels
        self.network = network

    def set_network(self, network):
        self.network = network

    def __len__(self):
        # this time the batch size = number of proteins = number of datapoints for the dataloader
        # For CNN:
        if self.network == "CNN":
            return len(self.labels)
        # For FNN:
        elif self.network == "FNN":
            return sum([len(protein) for protein in self.labels])

    def number_residues(self):
        return sum([len(protein) for protein in self.labels])

    def __getitem__(self, index):
        if self.network == "CNN":
            # I have to provide 3-dimensional input to conv1d, so proteins must be organised in batches
            try:
                return torch.tensor(self.inputs[index]).float(), torch.tensor(self.labels[index], dtype=torch.long)
            except IndexError:
                return None
        elif self.network == "FNN":
            k = 0  # k is the current protein index, index gets transformed to the position in the sequence
            protein_length = len(self.labels[k])
            while index >= protein_length:
                index = index - protein_length
                k += 1
                protein_length = len(self.labels[k])
            return torch.tensor(self.inputs[k][index]).float(), torch.tensor(self.labels[k][index])


class CNNSmall(nn.Module):
    def __init__(self):
        super().__init__()
        # version 0: 2 C layers
        self.conv1 = nn.Conv1d(in_channels=1025, out_channels=32, kernel_size=5, padding=2)
        # --> out: (32, proteins_length)
        # self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, padding=2)
        # --> out: (1, protein_length)

    def forward(self, input):
        # version 0: 2 C layers
        x = self.conv1(input.transpose(1, 2).contiguous())
        # x = self.dropout(x)   # dropout makes it worse...
        x = self.relu(x)
        x = self.conv2(x)
        x = x + 2
        return x


class CNNLarge(nn.Module):
    def __init__(self):
        super().__init__()
        # version 1: 5 C layers
        self.conv1 = nn.Conv1d(in_channels=1025, out_channels=512, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        # --> out: (1, protein_length)

    def forward(self, input):
        # version 1: 5 C layers
        x = self.conv1(input.transpose(1, 2).contiguous())
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        return x


class FNN(nn.Module):
    def __init__(self, input_size, output_size, p):
        super(FNN, self).__init__()
        self.input_layer = nn.Linear(input_size, input_size)
        self.hidden_layer = nn.Linear(input_size, int(input_size / 2))
        self.output_layer = nn.Linear(int(input_size / 2), output_size)
        self.dropout = nn.Dropout(p)

    def forward(self, input, multilabel):
        x = F.relu(self.input_layer(input))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer(x))
        x = self.dropout(x)
        if multilabel:
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


def conf_matrix(prediction, label, batch_wise_loss, batch_wise, fold, random):
    batch_wise_loss.append(loss_function(prediction, label.to(torch.float32)).item())
    # apply activation function to prediction to enable classification and transpose matrices
    if random:
        prediction_act = prediction     # no sigmoid needed for these values
    else:
        prediction_act = torch.sigmoid(prediction)
    prediction_max = prediction_act > cutoff[fold]

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
    return {"Precision": precision, "Recall": recall, "Neg_Precision": neg_precision, "Neg_Recall": neg_recall,
            "Balanced Acc.": balanced_acc, "F1": f1, "MCC": mcc}


def assess(name, cutoff, mode, multilabel, network, loss_function):
    # predict and assess performance of 1 model
    if multilabel:
        all_conf_matrices = [{"correct": [], "TP": [], "FP": [], "TN": [], "FN": []},
                             {"correct": [], "TP": [], "FP": [], "TN": [], "FN": []},
                             {"correct": [], "TP": [], "FP": [], "TN": [], "FN": []}]
    else:
        all_conf_matrices = {"correct": [], "TP": [], "FP": [], "TN": [], "FN": []}
    for fold, _ in enumerate(cutoff):
        print(f"{name} Fold {fold}")
        # for validation use the training IDs in the current fold

        # read target data y and disorder information
        # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
        val_labels = read_labels(fold, None)

        # create the input and target data exactly how it's fed into the ML model
        # and add the confounding feature of disorder to the embeddings
        this_fold_val_input, this_fold_val_target = get_ML_data(val_labels, embeddings, mode, multilabel, None)

        # instantiate the dataset
        validation_dataset = BindingDataset(this_fold_val_input, this_fold_val_target, network)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_size = 1025 if mode == 'all' else 1024
        output_size = 3 if multilabel else 1

        if network == "FNN":
            model = FNN(input_size, output_size, dropout).to(device)
        elif variant == 0.0:
            model = CNNSmall().to(device)
        elif variant == 1.0:
            model = CNNLarge().to(device)

        if name.startswith("random"):
            model = None
        else:
            model.load_state_dict(
                torch.load(f"../results/models/binding_regions_model_{name}_fold_{fold}.pth"))

        batch_size = 1 if network == "CNN" else 339  # 339 is avg protein length
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
                batch_wise = {"correct": [], "TP": [], "FP": [], "TN": [], "FN": []}

                for input, label in test_loader:
                    input, label = input.to(device), label[:, None].to(device)
                    if name.startswith("random"):
                        prediction = torch.rand(len(label), 1).to(device)
                        random = True
                    else:
                        prediction = model(input, multilabel) if network == "FNN" else model(input)
                        random = False
                    batch_wise_loss, batch_wise = conf_matrix(prediction, label, batch_wise_loss, batch_wise, fold, random)

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

        # calculate sum and avg values
        sum_matrix = [{}, {}, {}]
        for k in all_conf_matrices[0].keys():
            sum_matrix[0][k] = np.sum(all_conf_matrices[0][k])
            sum_matrix[1][k] = np.sum(all_conf_matrices[1][k])
            sum_matrix[2][k] = np.sum(all_conf_matrices[2][k])

        avg_metrics = [{}, {}, {}]
        for k in all_metrics[0].keys():
            avg_metrics[0][k] = np.average(all_metrics[0][k])
            avg_metrics[1][k] = np.average(all_metrics[1][k])
            avg_metrics[2][k] = np.average(all_metrics[2][k])


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

        # calculate sum and avg values
        sum_matrix = {}
        for k in all_conf_matrices.keys():
            sum_matrix[k] = np.sum(all_conf_matrices[k])

        avg_metrics = {}
        for k in all_metrics.keys():
            avg_metrics[k] = np.average(all_metrics[k])

    return sum_matrix, avg_metrics, all_sd_errors


if __name__ == '__main__':
    # read input embeddings
    embeddings_in = '../dataset/train_set.h5'
    embeddings = dict()
    with h5py.File(embeddings_in, 'r') as f:
        for key, embedding in f.items():
            original_id = embedding.attrs['original_id']
            embeddings[original_id] = np.array(embedding)
    # now {IDs: embeddings} are written in the embeddings dictionary

    variants = [0.0, 1.0, 2.0, 2.1, 2.20, 2.21, 12.0, 3.0, 13.0, 4.0, 4.1, 14.0]
    # cutoffs are different for each fold, variant (and class, if multiclass)!
    cutoffs = {0.0: [0.315, 0.16, 0.235, 0.245, 0.39],
               1.0: [0.005, 0.005, 0.005, 0.005, 0.01],
               2.0: [0.12, 0.1, 0.06, 0.06, 0.08],
               2.1: [0.05, 0.05, 0.15, 0.2, 0.85],
               2.20: [0.65, 0.65, 0.65, 0.5, 0.8],
               2.21: [0.8, 0.8, 0.85, 0.85, 0.85],
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
             12.0: "random_binary",
             3.0: "3_d_only",
             13.0: "random_d_only",
             4.0: "4_multiclass",
             4.1: "4-1_new_oversampling",
             14.0: "random_multilabel"}

    performances = []
    for variant in variants:
        # set parameters
        print(f"variant {variant}:")
        mode = 'disorder_only' if (variant % 10) == 3.0 else 'all'
        multilabel = True if (variant % 10) >= 4.0 else False
        network = 'CNN' if (variant % 10) < 2.0 else 'FNN'
        loss_function = nn.BCELoss() if multilabel else nn.BCEWithLogitsLoss()
        if variant == 2.20:
            dropout = 0.2
        elif variant == 2.21:
            dropout = 0.3
        else:
            dropout = 0.0
        cutoff = cutoffs[variant]
        name = names[variant]

        performances.append(assess(name, cutoff, mode, multilabel, network, loss_function))

    with open('../results/logs/performance_assessment.tsv', "w") as output:
        output.write("model\tclass\t")
        for key in performances[0][0].keys():  # conf-matrix
            output.write(str(key) + "\t")
        for key in performances[0][1].keys():  # metrics
            output.write(str(key) + "\t")
        for key in performances[0][2].keys():  # SEs of metrics
            output.write("SE_" + str(key) + "\t")
        output.write("\n")

        for i, v in enumerate(variants):
            if (v % 10) >= 4.0:  # multiclass
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

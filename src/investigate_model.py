
from src import CV_splits
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
    if oversampling is None:        # no oversampling on validation set!
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
                if emb_with_conf.shape[0] != len(labels[id][1]):    # sanity check
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
            binding_encoded[1] = list(re.sub(r'N|X|Z|A', '1', re.sub(r'-|_|P|O|Y', '0', binding)))  # nucleic-acid-binding?
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
    def __init__(self, embeddings, binding_labels):
        self.inputs = embeddings
        self.labels = binding_labels

    def __len__(self):
        # this time the batch size = number of proteins = number of datapoints for the dataloader
        # For CNN:
        # return len(self.labels)
        # For FNN:
        return sum([len(protein) for protein in self.labels])

    def number_residues(self):
        return sum([len(protein) for protein in self.labels])

    # for CNN
    """
    def __getitem__(self, index):
        # I have to provide 3-dimensional input to conv1d, so proteins must be organised in batches
        try:
            return torch.tensor(self.inputs[index]).float(), torch.tensor(self.labels[index], dtype=torch.long)
        except IndexError:
            return None
    """
    # for FNN
    def __getitem__(self, index):
        k = 0  # k is the current protein index, index gets transformed to the position in the sequence
        protein_length = len(self.labels[k])
        while index >= protein_length:
            index = index - protein_length
            k += 1
            protein_length = len(self.labels[k])
        return torch.tensor(self.inputs[k][index]).float(), torch.tensor(self.labels[k][index])



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        """
        # version 0: 2 C layers
        self.conv1 = nn.Conv1d(in_channels=1025, out_channels=32, kernel_size=5, padding=2)
        # --> out: (32, proteins_length)
        # self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, padding=2)
        # --> out: (1, protein_length)
        """
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
        """
        # version 0: 2 C layers
        x = self.conv1(input.transpose(1, 2).contiguous())
        # x = self.dropout(x)   # dropout makes it worse...
        x = self.relu(x)
        x = self.conv2(x)
        x = x+2
        """
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
            output = self.output_layer(x)   # rather without sigmoid to apply BCEWithLogitsLoss later
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


if __name__ == '__main__':
    # read input embeddings
    embeddings_in = '../dataset/train_set.h5'
    embeddings = dict()
    with h5py.File(embeddings_in, 'r') as f:
        for key, embedding in f.items():
            original_id = embedding.attrs['original_id']
            embeddings[original_id] = np.array(embedding)
    # now {IDs: embeddings} are written in the embeddings dictionary

    def try_cutoffs(mode, multilabel):
        # iterate over folds
        with open(f"../results/logs/validation_2_2_dropout_{dropout}_lr_0.008.txt", "w") as output_file:
            if multilabel:
                output_file.write('Fold\tAvg_Loss\tCutoff\tP_Acc\tP_Prec\tP_Rec\tP_TP\tP_FP\tP_TN\tP_FN\t'
                                  'N_Acc\tN_Prec\tN_Rec\tN_TP\tN_FP\tN_TN\tN_FN\t'
                                  'O_Acc\tO_Prec\tO_Rec\tO_TP\tO_FP\tO_TN\tO_FN\n')
            else:
                output_file.write('Fold\tAvg_Loss\tCutoff\tAcc\tPrec\tRec\tTP\tFP\tTN\tFN\n')
            for fold in range(5):
                print("Fold: " + str(fold))
                # for validation use the training IDs in the current fold

                # read target data y and disorder information
                # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
                val_labels = read_labels(fold, None)    # no oversampling on validation labels

                # create the input and target data exactly how it's fed into the ML model
                # and add the confounding feature of disorder to the embeddings
                this_fold_val_input, this_fold_val_target = get_ML_data(val_labels, embeddings, mode, multilabel, None)

                # instantiate the dataset
                validation_dataset = BindingDataset(this_fold_val_input, this_fold_val_target)


                """
                # look at some data:
                for i in range(50, 54):
                   input, label = training_dataset[i]
                   print(f'Embedding input:\n {input}\nPrediction target:\n{label}\n\n')
                """

                def criterion(loss_func, prediction, label):  # sum over all classification heads
                    losses = 0
                    prediction = prediction.T
                    label = label.T
                    for i, _ in enumerate(prediction):  # for each class (-> 1-dimensional loss)
                        losses += loss_func(prediction[i], label[i])
                    return losses

                def test_performance(dataset, model, loss_function, device, output, multilabel):
                    batch_size = 512
                    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
                    size = dataset.number_residues()
                    # print(size)
                    model.eval()
                    with torch.no_grad():
                        # try out different cutoffs
                        for cutoff in (0.01 * np.arange(0, 100, step=5)):   # mult. works around floating point precision issue
                            if multilabel:
                                test_loss = 0
                                p_correct, p_tp, p_fp, p_tn, p_fn = 0, 0, 0, 0, 0
                                n_correct, n_tp, n_fp, n_tn, n_fn = 0, 0, 0, 0, 0
                                o_correct, o_tp, o_fp, o_tn, o_fn = 0, 0, 0, 0, 0
                                for input, label in test_loader:
                                    input, label = input.to(device), label.to(device)
                                    prediction = model(input, multilabel)
                                    test_loss += criterion(loss_function, prediction, label.to(torch.float32)).item()
                                    # apply activation function to prediction to enable classification and transpose matrices
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
                                    print(f"cutoff {cutoff}\tProtein-binding:\tAccuracy: {(100 * p_correct):>0.1f}%, Sensitivity: {(100 * (p_tp / (p_tp + p_fn))): >0.1f}%, Precision: {(100 * (p_tp / (p_tp + p_fp))): >0.1f}%, Avg loss: {test_loss:>8f}")
                                    output.write('\t'.join(
                                        [str(fold), str(round(test_loss, 6)), str(cutoff),
                                         str(round(100 * p_correct, 1)), str(round(100 * (p_tp / (p_tp + p_fp)), 1)), str(round(100 * (p_tp / (p_tp + p_fn)), 1)),
                                         str(p_tp), str(p_fp), str(p_tn), str(p_fn)]) + '\t')
                                except ZeroDivisionError:
                                    print(f"cutoff {cutoff}\tProtein-binding:\tAccuracy: {(100 * p_correct):>0.1f}%, Sensitivity: 0.0%, Precision: 0.0%, Avg loss: {test_loss:>8f}")
                                    output.write('\t'.join(
                                        [str(fold), str(round(test_loss, 6)), str(cutoff),
                                         str(round(100 * p_correct, 1)), '0.0', '0.0',
                                         str(p_tp), str(p_fp), str(p_tn), str(p_fn)]) + '\t')
                                try:
                                    print(f"cutoff {cutoff}\tNuc-binding:\tAccuracy: {(100 * n_correct):>0.1f}%, Sensitivity: {(100 * (n_tp / (n_tp + n_fn))): >0.1f}%, Precision: {(100 * (n_tp / (n_tp + n_fp))): >0.1f}%, Avg loss: {test_loss:>8f}")
                                    output.write('\t'.join(
                                        [str(round(100 * n_correct, 1)), str(round(100 * (n_tp / (n_tp + n_fp)), 1)), str(round(100 * (n_tp / (n_tp + n_fn)), 1)),
                                         str(n_tp), str(n_fp), str(n_tn), str(n_fn)]) + '\t')
                                except ZeroDivisionError:
                                    print(f"cutoff {cutoff}\tNuc-binding:\tAccuracy: {(100 * n_correct):>0.1f}%, Sensitivity: 0.0%, Precision: 0.0%, Avg loss: {test_loss:>8f}")
                                    output.write('\t'.join(
                                        [str(round(100 * n_correct, 1)), '0.0', '0.0',
                                         str(n_tp), str(n_fp), str(n_tn), str(n_fn)]) + '\t')
                                try:
                                    print(f"cutoff {cutoff}\tOther-binding:\tAccuracy: {(100 * o_correct):>0.1f}%, Sensitivity: {(100 * (o_tp / (o_tp + o_fn))): >0.1f}%, Precision: {(100 * (o_tp / (o_tp + o_fp))): >0.1f}%, Avg loss: {test_loss:>8f}\n")
                                    output.write('\t'.join(
                                        [str(round(100 * o_correct, 1)), str(round(100 * (o_tp / (o_tp + o_fp)), 1)), str(round(100 * (o_tp / (o_tp + o_fn)), 1)),
                                         str(o_tp), str(o_fp), str(o_tn), str(o_fn)]) + '\n')
                                except ZeroDivisionError:
                                    print(f"cutoff {cutoff}\tOther-binding:\tAccuracy: {(100 * o_correct):>0.1f}%, Sensitivity: 0.0%, Precision: 0.0%, Avg loss: {test_loss:>8f}\n")
                                    output.write('\t'.join(
                                        [str(round(100 * o_correct, 1)), '0.0', '0.0',
                                         str(o_tp), str(o_fp), str(o_tn), str(o_fn)]) + '\n')

                            else:    # not multilabel
                                test_loss, correct, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0
                                for input, label in test_loader:
                                    input, label = input.to(device), label[:, None].to(device)
                                    prediction = model(input, multilabel)
                                    test_loss += loss_function(prediction, label.to(torch.float32)).item()
                                    # apply activation function to prediction to enable classification
                                    prediction_act = torch.sigmoid(prediction)
                                    prediction_max = prediction_act > cutoff
                                    # metrics
                                    correct += (prediction_max == label).type(torch.float).sum().item()
                                    tp += (prediction_max == label)[label == 1].type(torch.float).sum().item()
                                    fp += (prediction_max != label)[label == 0].type(torch.float).sum().item()
                                    tn += (prediction_max == label)[label == 0].type(torch.float).sum().item()
                                    fn += (prediction_max != label)[label == 1].type(torch.float).sum().item()

                                test_loss /= int(size/batch_size)
                                correct /= size
                                try:
                                    print(
                                    f"cutoff {cutoff}\tAccuracy: {(100 * correct):>0.1f}%, Sensitivity: {(100 * (tp / (tp+fn))): >0.1f}%, Precision: {(100 * (tp / (tp+fp))): >0.1f}%, Avg loss: {test_loss:>8f}")
                                    output.write('\t'.join([str(fold), str(round(test_loss, 6)), str(cutoff), str(round(100 * correct, 1)),
                                                        str(round(100 * (tp / (tp+fp)), 1)), str(round(100 * (tp / (tp+fn)),1)),
                                                        str(tp), str(fp), str(tn), str(fn)]) + '\n')
                                except ZeroDivisionError:
                                    print(
                                        f"cutoff {cutoff}\tAccuracy: {(100 * correct):>0.1f}%, Sensitivity: 0.0%, Precision: 0.0%, Avg loss: {test_loss:>8f}")
                                    output.write('\t'.join(
                                        [str(fold), str(round(test_loss, 6)), str(cutoff), str(round(100 * correct, 1)),
                                         '0.0', '0.0', str(tp), str(fp), str(tn), str(fn)]) + '\n')


                device = "cuda" if torch.cuda.is_available() else "cpu"
                input_size = 1024 if mode == 'disorder_only' else 1025
                output_size = 3 if multilabel else 1
                model = FNN(input_size=input_size, output_size=output_size, p=dropout).to(device)
                model.load_state_dict(
                    torch.load(f"../results/models/binding_regions_model_2-2_dropout_{dropout}_lr_0.008_fold_{fold}.pth"))
                # test performance again, should be the same
                loss_function = nn.BCELoss() if multilabel else nn.BCEWithLogitsLoss()
                test_performance(validation_dataset, model, loss_function, device, output_file, multilabel)

    def predictCNN(cutoff, fold):
        with open(f"../results/logs/predict_val_1_5layers_{fold}_{cutoff}.txt", "w") as output_file:
            print("Fold: " + str(fold))
            # for validation use the training IDs in the current fold

            # read target data y and disorder information
            # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
            val_labels = read_labels(fold, oversampling)
            ids = list(val_labels.keys())

            # create the input and target data exactly how it's fed into the ML model
            # and add the confounding feature of disorder to the embeddings
            this_fold_val_input, this_fold_val_target = get_ML_data(val_labels, embeddings)

            # instantiate the dataset
            validation_dataset = BindingDataset(this_fold_val_input, this_fold_val_target)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CNN().to(device)
            model.load_state_dict(
                torch.load(f"../results/models/binding_regions_model_1_5layers_fold_{fold}.pth"))
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

                    output_file.write(f'{ids[i]}\nlabels:\t{label}\nprediction_0:\t{prediction_act}\nprediction_1:\t{prediction_max}\n')

    class Zone:
        def __init__(self, start: int, end: int, value: float, last: bool):
            # start incl, end excl.
            self.length = end - start
            self.value = value
            # identify specific zones in prediction:
            # pos_short: positive, len < 5, not at the end
            # neg_short: negative, len < 10, not at the start or end
            # pos_middle: positive, not short, len < 55
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
                    self.type = "pos_middle"
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
        # pos_middle: positive, not short, len < 55
        # pos_long: positive, 55 <= len <= 240
        # pos_valid: positive, len > 240
        # neg_valid: negative, len >= 10
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
        # 2. pos_middle, neg_short, pos_middle --> 0s, (0s), 0s
        # 3. pos_middle, neg_short, pos_long/pos_valid --> 0s, (0s), (1s)
        # 4. pos_long/pos_valid, neg_short, pos_middle --> (1s), (0s), 0s
        # 5. pos_long/pos_valid, neg_short, pos_long/pos_valid --> (1s), 1s, (1s)
        # 6. pos_short, neg_short, pos_middle/pos_long --> (0s), (0s), 0s
        # 7. pos_middle/pos_long, neg_short, pos_short --> 0s, (0s), (0s)
        for zone in zones:
            pass




    def predictFNN(cutoff, cutoff_p, cutoff_n, cutoff_o, fold, mode, multilabel, post_processing):
        with open(f"../results/logs/predict_val_2-2_dropout_{dropout}_new_{fold}_{cutoff}.txt", "w") as output_file:
            print("Fold: " + str(fold))
            # for validation use the training IDs in the current fold

            # read target data y and disorder information
            # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
            val_labels = read_labels(fold, None)

            # create the input and target data exactly how it's fed into the ML model
            # and add the confounding feature of disorder to the embeddings
            this_fold_val_input, this_fold_val_target = get_ML_data(val_labels, embeddings, mode, multilabel, None)

            # instantiate the dataset
            validation_dataset = BindingDataset(this_fold_val_input, this_fold_val_target)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_size = 1025 if mode == 'all' or multilabel else 1024
            output_size = 3 if multilabel else 1
            model = FNN(input_size, output_size, dropout).to(device)
            model.load_state_dict(
                torch.load(f"../results/models/binding_regions_model_2-2_dropout_{dropout}_new_fold_{fold}.pth"))
            # test performance again, should be the same

            test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=512, shuffle=False)
            model.eval()
            all_prediction_act = list()
            all_prediction_max = list()
            all_labels = list()
            with torch.no_grad():
                for i, (input, label) in enumerate(test_loader):
                    if multilabel:
                        input, label = input.to(device), label.to(device).T
                        label = [transform_output(
                            int(label[0][i]), int(label[1][i]), int(label[2][i]))
                            for i, _ in enumerate(label[0])]
                        all_labels.extend(label)
                        prediction = model(input, multilabel).T
                        # apply activation function to prediction to enable classification
                        prediction_max_p = prediction[0] > cutoff_p
                        prediction_max_n = prediction[1] > cutoff_n
                        prediction_max_o = prediction[2] > cutoff_o
                        prediction_max = [transform_output(
                            int(prediction_max_p[i]), int(prediction_max_n[i]), int(prediction_max_o[i]))
                            for i, _ in enumerate(prediction_max_p)]
                        all_prediction_max.extend(prediction_max)
                    else:
                        input, label = input.to(device), label[:, None].to(device)
                        all_labels.extend(label.flatten().tolist())
                        prediction = model(input, multilabel)
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
                    output_file.write(f'\nprediction_1:\t{torch.tensor(all_prediction_max[delimiter_0: delimiter_1])}\n')
                    if post_processing:
                        output_file.write(f'\nprediction_pp:\t{post_process(torch.tensor(all_prediction_max[delimiter_0 : delimiter_1]))}\n')

                delimiter_0 = delimiter_1


    oversampling = 'binary_residues'     # binary, binary_residues or multiclass_residues
    mode = 'all'  # disorder_only or all
    multilabel = False
    dropout = 0.3
    # try_cutoffs(mode=mode, multilabel=multilabel)  # expensive!

    # get predictions for chosen cutoff, fold
    cutoff = 0.85
    cutoff_p, cutoff_n, cutoff_o = 0.3, 0.45, 0.25
    fold = 4
    post_processing = True
    # predictCNN(cutoff, fold, mode)
    predictFNN(cutoff, cutoff_p, cutoff_n, cutoff_o, fold, mode, multilabel, post_processing)




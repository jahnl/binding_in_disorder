"""
starting point for predictor development:
simple FNN, binary prediction of binding/non-binding,
given embeddings + confounding feature: disorder
labels: anything vs '_' in 4th line of labels per protein
variant 0: all residues, variant 1: only disordered residues
"""
from torch.autograd.grad_mode import F

from src import CV_splits
import numpy as np
import h5py
from Bio import SeqIO
import re
import torch.tensor
from torch.utils.data import Dataset
from torch import nn, optim
import torch.nn.functional as F
import datetime
import copy


def read_labels(fold, oversampling):
    with open(f'../dataset/folds/CV_fold_{fold}_labels_{oversampling}.txt') as handle:
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
            if record.id == 'Q98157':
                print(record.id, labels[record.id])
            """
    return labels


def get_ML_data(labels, embeddings, mode):
    input = list()
    target = list()
    for id in labels.keys():
        if mode == 'all':
            conf_feature = str(labels[id][1])
            conf_feature = list(conf_feature.replace('-', '0').replace('D', '1'))
            conf_feature = np.array(conf_feature, dtype=float)
            emb_with_conf = np.column_stack((embeddings[id], conf_feature))
            input.append(emb_with_conf)
        elif mode == 'disorder_only':
            bool_list = [False if x == '-' else True for x in list(labels[id][2])]
            input.append(embeddings[id][bool_list])
        # for target: 0 = non-binding, 1 = binding, 0 = not in disordered region (2 doesnt work!, would be multi-class)
        binding = str(labels[id][2])
        if mode == 'all':
            binding = re.sub(r'-|_', '0', binding)
        elif mode == 'disorder_only':
            binding = binding.replace('-', '').replace('_', '0')
        binding = list(re.sub(r'P|N|O|X|Y|Z|A', '1', binding))
        binding = np.array(binding, dtype=float)
        target.append(binding)

        """
        if id == 'Q98157':
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
        return sum([len(protein) for protein in self.labels])

    def number_residues(self):
        return len(self)


    def __getitem__(self, index):
        k = 0  # k is the current protein index, index gets transformed to the position in the sequence
        protein_length = len(self.labels[k])
        while index >= protein_length:
            index = index - protein_length
            k += 1
            protein_length = len(self.labels[k])
        return torch.tensor(self.inputs[k][index]).float(), torch.tensor(self.labels[k][index])




class FNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FNN, self).__init__()
        self.input_layer = nn.Linear(input_size, input_size)
        self.hidden_layer = nn.Linear(input_size, int(input_size / 2))
        self.output_layer = nn.Linear(int(input_size / 2), output_size)

    def forward(self, input):
        x = F.relu(self.input_layer(input))
        x = F.relu(self.hidden_layer(x))
        # output = torch.sigmoid(self.output_layer(x))
        output = self.output_layer(x)   # rather without sigmoid to apply BCEWithLogitsLoss later
        return output


if __name__ == '__main__':
    # apply cross-validation and oversampling on training dataset
    oversampling = 'binary'
    #CV_splits.split(oversampling)

    mode = 'disorder_only'  # disorder_only or all

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
        # for training use all training IDs except for the ones in the current fold.
        # for validation use the training IDs in the current fold

        # read target data y and disorder information
        # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
        val_labels = read_labels(fold, oversampling)
        train_labels = {}
        for train_fold in range(5):     # TODO: link number of files to config file
            if train_fold != fold:
                train_labels.update(read_labels(train_fold, oversampling))
        print(len(val_labels), len(train_labels))

        # create the input and target data exactly how it's fed into the ML model
        # and add the confounding feature of disorder to the embeddings
        this_fold_input, this_fold_target = get_ML_data(train_labels, embeddings, mode)
        this_fold_val_input, this_fold_val_target = get_ML_data(val_labels, embeddings, mode)

        # instantiate the dataset
        training_dataset = BindingDataset(this_fold_input, this_fold_target)
        validation_dataset = BindingDataset(this_fold_val_input, this_fold_val_target)

        """
        # look at some data:
        for i in range(50, 54):
           input, label = training_dataset[i]
           print(f'Embedding input:\n {input}\nPrediction target:\n{label}\n\n')
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device: " + device)
        input_size = 1024 if mode == 'disorder_only' else 1025
        model = FNN(input_size=input_size, output_size=1).to(device)
        criterion = nn.BCEWithLogitsLoss()    # loss function for binary problem
        optimizer = optim.Adam(model.parameters(), lr=0.01)


        def train(dataset, model, loss_function, optimizer, device, output):
            avg_train_loss = 0
            batch_size = 512
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            nr_samples = dataset.number_residues()
            for i, (input, label) in enumerate(train_loader):
                input, label = input.to(device), label[:, None].to(device)  # make sure both have same dimensions

                # make a prediction
                prediction = model(input)
                # compute loss
                loss = loss_function(prediction, label.to(torch.float32))

                # if i == 260:
                #    print(f'protein 260: prediction: {prediction, prediction.dtype}')
                #    print(f'protein 260: label:      {label, label.dtype}')
                #    print(f'protein 260: loss:       {loss}')

                avg_train_loss += loss.item()

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print(f'\tLoss: {loss.item()} \t batch:{i}/{int(nr_samples / batch_size)}')
            avg_train_loss /= int(nr_samples / batch_size)
            print("\tAvg_train_loss: " + str(avg_train_loss))
            output.write(f"\tTraining set: Avg loss: {avg_train_loss:>8f} \n")


        def test_performance(dataset, model, loss_function, device, output):
            batch_size = 512
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            size = dataset.number_residues()
            # print(size)
            model.eval()
            test_loss, correct, tp, tp_fn, tp_fp = 0, 0, 0, 0, 0
            with torch.no_grad():
                for input, label in test_loader:
                    input, label = input.to(device), label[:, None].to(device)
                    prediction = model(input)
                    test_loss += loss_function(prediction, label.to(torch.float32)).item()
                    # apply activation function to prediction to enable classification
                    prediction_act = torch.sigmoid(prediction)
                    # prediction_max = prediction_act.argmax(1)     # argmax only if its multi-class
                    prediction_max = prediction_act > 0.3
                    # metrics
                    correct += (prediction_max == label).type(torch.float).sum().item()
                    tp += (prediction_max == label)[label == 1].type(torch.float).sum().item()
                    tp_fn += (label == 1).type(torch.float).sum().item()
                    tp_fp += (prediction_max == 1).type(torch.float).sum().item()

                    """
                    if correct < 398:
                        # print(f'val_prediction: {prediction}')
                        print(f'val_prediction_activated: {prediction_act}')
                        # print(f'val_prediction_max: {prediction_max}')
                        print(f'val_labels: {label}')
                        print(f'loss: {test_loss}')
                        print(f'correct: {correct}')
                        print(f'tp: {tp}, tp+fn: {tp_fn}, tp+fp: {tp_fp}')
                    """



                test_loss /= int(size/batch_size)
                correct /= size
                try:
                    print(f"\tAccuracy: {(100 * correct):>0.1f}%, Sensitivity: {(100 * (tp/tp_fn)): >0.1f}%, "
                          f"Precision: {(100 * (tp/tp_fp)): >0.1f}%, Avg loss: {test_loss:>8f} \n")
                    output.write(f"\tCross-Training set: Accuracy: {(100 * correct):>0.1f}%, "
                                 f"Sensitivity: {(100 * (tp / tp_fn)): >0.1f}%, "
                                 f"Precision: {(100 * (tp / tp_fp)): >0.1f}%, Avg loss: {test_loss:>8f} \n")
                except ZeroDivisionError:
                    print(f"\tAccuracy: {(100 * correct):>0.1f}%, Sensitivity: NA, Precision: NA, "
                          f"Avg loss: {test_loss:>8f} \n")
                    output.write(f"\tCross-Training set: Accuracy: {(100 * correct):>0.1f}%, "
                                 f"Sensitivity: NA%, Precision: NA, Avg loss: {test_loss:>8f} \n")

            return test_loss

        # initialize training parameters for early stopping
        epochs = 200
        min_val_loss = np.Inf
        epochs_no_improvement = 0
        n_epochs_stop = 10
        best_state_dict = None

        output_file = open("../results/logs/training_progress_3_d_only_fold_" + str(fold) + ".txt", "w")

        for epoch in range(epochs):
            print(f'{datetime.datetime.now()}\tEpoch {epoch + 1}')
            output_file.write(f'{datetime.datetime.now()}\tEpoch {epoch + 1}\n')
            train(training_dataset, model, criterion, optimizer, device, output_file)
            test_loss = test_performance(validation_dataset, model, criterion, device, output_file)

            # early stopping
            if test_loss < min_val_loss:    # improvement
                min_val_loss = test_loss
                epochs_no_improvement = 0
                best_state_dict = copy.deepcopy(model.state_dict())

            else:                           # no improvement
                epochs_no_improvement += 1
                if epochs_no_improvement == n_epochs_stop:
                    break   # -> early stopping



        # save best model of this fold
        torch.save(best_state_dict, "../results/models/binding_regions_model_3_d_only_fold_" + str(fold) + ".pth")
        output_file.flush()
        output_file.close()

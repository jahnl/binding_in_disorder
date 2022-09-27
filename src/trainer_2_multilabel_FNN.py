"""
simple FNN, multi-class prediction of non-binding, protein-binding, nuc-binding, other-binding, or a combination of those
given embeddings + confounding feature: disorder
labels: anything in 4th line of labels per protein, but one-hot encoded
"""
from torch.autograd.grad_mode import F

from src import CV_and_oversampling
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
            """
            if record.id == 'E2IHW6*':
                print(record.id, labels[record.id], record)
            """
    return labels


def get_ML_data(labels, embeddings, new_datapoints):
    input = list()
    target = list()
    datapoint_counter = 0
    for id in labels.keys():

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
    def __init__(self, input_size, output_size, p):
        super(FNN, self).__init__()
        self.input_layer = nn.Linear(input_size, input_size)
        self.hidden_layer = nn.Linear(input_size, int(input_size / 2))
        self.output_layer = nn.Linear(int(input_size / 2), output_size)
        self.dropout = nn.Dropout(p)

    def forward(self, input):
        x = F.relu(self.input_layer(input))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer(x))
        x = self.dropout(x)
        output = torch.sigmoid(self.output_layer(x))
        return output


if __name__ == '__main__':
    # apply cross-validation and oversampling on training dataset
    oversampling = 'multiclass_residues'
    #CV_and_oversampling.split(n_splits, oversampling)

    dropout = 0

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
        # for validation use the training IDs in the current fold, but without oversampling

        # read target data y and disorder information
        # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
        val_labels = read_labels(fold, None)    # no oversampling on val_labels
        train_labels = {}
        for train_fold in range(5):     # TODO: link number of files to config file
            if train_fold != fold:
                train_labels.update(read_labels(train_fold, oversampling))
        print(len(val_labels), len(train_labels))

        # load pre-computed datapoint embeddings
        t_datapoints = list()
        if 'residues' in oversampling:
            for f in range(5):
                if f != fold:
                    t_datapoints.extend(np.load(f'../dataset/folds/new_datapoints_{oversampling}_fold_{f}.npy', allow_pickle=True))

        # create the input and target data exactly how it's fed into the ML model
        # and add the confounding feature of disorder to the embeddings
        this_fold_input, this_fold_target = get_ML_data(train_labels, embeddings, t_datapoints)
        this_fold_val_input, this_fold_val_target = get_ML_data(val_labels, embeddings, None)

        # instantiate the dataset
        training_dataset = BindingDataset(this_fold_input, this_fold_target)
        validation_dataset = BindingDataset(this_fold_val_input, this_fold_val_target)

        """
        # look at some data:
        for i in range(50, 54):
           input, label = training_dataset[i]
           print(f'Embedding input:\n {input}\nPrediction target:\n{label}\n\n')
        """

        def criterion(loss_func, prediction, label):    # sum over all classification heads
            losses = 0
            prediction = prediction.T
            label = label.T
            for i, _ in enumerate(prediction):     # for each class (-> 1-dimensional loss)
                losses += loss_func(prediction[i], label[i])
            return losses

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device: " + device)
        input_size = 1025
        model = FNN(input_size=input_size, output_size=3, p=dropout).to(device)
        # loss_function = nn.CrossEntropyLoss()
        loss_function = nn.BCELoss()    # not multi-class when applied to each dimension individually
        optimizer = optim.Adam(model.parameters(), lr=0.01)


        def train(dataset, model, loss_function, optimizer, device, output):
            avg_train_loss = 0
            batch_size = 512
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            nr_samples = dataset.number_residues()
            for i, (input, label) in enumerate(train_loader):
                input, label = input.to(device), label.to(device)

                # make a prediction
                prediction = model(input)
                # compute loss
                loss = criterion(loss_function, prediction, label.to(torch.float32))
                # loss = loss_function(prediction, label.to(torch.long))    # not suitable for multi-label

                """
                if i == 50:
                    torch.set_printoptions(threshold=10_000)
                    print(f'batch {i}: prediction: \n{prediction.T, prediction.dtype}')
                    print(f'batch {i}: label:      \n{label.T, label.dtype}')
                    print(f'batch {i}: loss:       {loss}')
                """

                avg_train_loss += loss.item()

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 500 == 0:
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
                    input, label = input.to(device), label.to(device)
                    prediction = model(input)
                    test_loss += loss_function(prediction, label.to(torch.float32)).item()
                    # apply activation function to prediction to enable classification
                    prediction_act = torch.sigmoid(prediction)
                    # prediction_max = prediction_act.argmax(1)     # argmax only if its multi-class
                    prediction_max = prediction_act > 0.5
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

        output_file = open(f"../results/logs/training_progress_4-1_new_oversampling_fold_{fold}.txt", "w")

        for epoch in range(epochs):
            print(f'{datetime.datetime.now()}\tEpoch {epoch + 1}')
            output_file.write(f'{datetime.datetime.now()}\tEpoch {epoch + 1}\n')
            train(training_dataset, model, loss_function, optimizer, device, output_file)
            test_loss = test_performance(validation_dataset, model, loss_function, device, output_file)

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
        torch.save(best_state_dict, f"../results/models/binding_regions_model_4-1_new_oversampling_fold_{fold}.pth")
        output_file.flush()
        output_file.close()

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
        # for target: 0 = non-binding, 1 = binding, 0 = not in disordered region (2 doesnt work!)
        binding = str(labels[id][2])
        binding = re.sub(r'-|_', '0', binding)
        binding = list(re.sub(r'P|N|O|X|Y|Z|A', '1', binding))
        binding = np.array(binding, dtype=float)
        target.append(binding)
        if id == 'Q98157':
            print(conf_feature)
            print(emb_with_conf.shape)
            print(binding)
    return input, target


# build the dataset
class BindingDataset(Dataset):
    def __init__(self, embeddings, binding_labels):
        self.inputs = embeddings
        self.labels = binding_labels

    def __len__(self):
        # this time the batch size = number of proteins = number of datapoints for the dataloader
        return len(self.labels)

    def number_residues(self):
        return sum([len(protein) for protein in self.labels])


    def __getitem__(self, index):
        #k = 0  # k is the current protein index, index gets transformed to the position in the sequence
        #protein_length = len(self.labels[k])
        #while index >= protein_length:
        #    index = index - protein_length
        #    k += 1
        #    protein_length = len(self.labels[k])
        #return torch.tensor(self.inputs[k][index]).float(), torch.tensor(self.labels[k][index])

        # I have to provide 3-dimensional input to conv1d, so proteins must be organised in batches
        try:
            return torch.tensor(self.inputs[index]).float(), torch.tensor(self.labels[index], dtype=torch.long)
        except IndexError:
            return None




class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv1d(1025, 512, 1025, kernel_size=5, padding=2)
        self.conv1 = nn.Conv1d(in_channels=1025, out_channels=32, kernel_size=5, padding=2)
        # --> out: (32, proteins_length)
        self.relu = nn.ReLU()   # self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, padding=2)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # --> out: (1, protein_length)
            # self.fc1 = nn.Linear(16 * 5 * 5, 120)
            # self.fc2 = nn.Linear(120, 84)
            # self.fc3 = nn.Linear(84, output_size)


    def forward(self, input):
        x = self.conv1(input.transpose(1, 2).contiguous())
        x = self.relu(x)
        x = self.conv2(x)
        x = x+2
        # x = self.softmax(x)
        # x = self.sigmoid(x)
        # x = self.relu(x)
        #x = x.view(-1, 16 * 5 * 5)
        return x


if __name__ == '__main__':
    # apply cross-validation and oversampling on training dataset
    oversampling = 'binary'
    CV_splits.split(oversampling)

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
        val_labels = read_labels(fold, oversampling)
        train_labels = {}
        for train_fold in range(5):     # TODO: link number of files to config file
            if train_fold != fold:
                train_labels.update(read_labels(train_fold, oversampling))
        print(len(val_labels), len(train_labels))

        # create the input and target data exactly how it's fed into the ML model
        # and add the confounding feature of disorder to the embeddings
        this_fold_input, this_fold_target = get_ML_data(train_labels, embeddings)
        this_fold_val_input, this_fold_val_target = get_ML_data(val_labels, embeddings)

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
        model = CNN().to(device)  # parameter = output size
        criterion = nn.BCEWithLogitsLoss()    # loss function for binary problem
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # TODO: tune these parameters


        def train(dataset, model, loss_function, optimizer, device, output):
            avg_train_loss = 0
            batch_size = 1
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            nr_samples = len(dataset)
            for i, (input, label) in enumerate(train_loader):
                input, label = input.to(device), label[None, :].to(device)  # ensure both have same dimensions

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

                # if i % 500 == 0:
                #    print(f'\tLoss: {loss.item()} \t batch:{i}/{int(nr_samples / batch_size)}')
            avg_train_loss /= int(nr_samples / batch_size)
            print("\tAvg_train_loss: " + str(avg_train_loss))
            output.write(f"\tTraining set: Avg loss: {avg_train_loss:>8f} \n")


        def test_performance(dataset, model, loss_function, device, output):
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            size = dataset.number_residues()
            # print(size)
            model.eval()
            test_loss, correct = 0, 0
            with torch.no_grad():
                for input, label in test_loader:
                    input, label = input.to(device), label[None, :].to(device)
                    prediction = model(input)
                    test_loss += loss_function(prediction, label.to(torch.float32)).item()

                    if correct == 0:
                        print(f'val_prediction: {prediction}')
                        # print(f'val_labels: {label}')
                        print(f'loss: {test_loss}')
                        print(f'correct: {(prediction.argmax(1) == label).type(torch.float).sum().item()}')

                    correct += (prediction.argmax(1) == label).type(torch.float).sum().item()

                test_loss /= size
                correct /= size
                print(f"\tAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
                output.write(f"\tCross-Training set: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            return test_loss

        # initialize training parameters for early stopping
        epochs = 200
        min_val_loss = np.Inf
        epochs_no_improvement = 0
        n_epochs_stop = 5
        best_state_dict = None

        output_file = open("../results/logs/training_progress_0_simple_fold_" + str(fold) + ".txt", "w")

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
        torch.save(best_state_dict, "../results/models/binding_regions_model_0_simple_fold_" + str(fold) + ".pth")
        output_file.flush()
        output_file.close()





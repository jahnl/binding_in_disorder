"""
starting point for predictor development:
simple model, binary prediction of binding/non-binding,
given embeddings + confounding feature: disorder
labels: anything vs '_' in 4th line of labels per protein
"""

from src import CV_and_oversampling
import numpy as np
import h5py
from Bio import SeqIO
import re
import torch.tensor
from torch.utils.data import Dataset
from torch import nn, optim
import datetime
import copy


def read_labels(fold, oversampling, dataset_dir):
    with open(f'{dataset_dir}folds/CV_fold_{fold}_labels_{oversampling}.txt', 'r') as handle:
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
            # for target: 0 = non-binding, 1 = binding, 0 = not in disordered region (2 doesn't work!)
            binding = str(labels[id][2])
            binding = re.sub(r'-|_', '0', binding)
            binding = list(re.sub(r'B|P|N|O|X|Y|Z|A', '1', binding))
            binding = np.array(binding, dtype=float)
            target.append(binding)
            """
            if id == 'Q98157':
                print(conf_feature)
                print(emb_with_conf.shape)
                print(binding)
            """
        elif mode == 'disorder_only':
            # separate regions of the same protein from each other!
            bool_list = [False if x == '-' else True for x in list(labels[id][2])]
            starts = []
            stops = []
            current = bool_list[0]
            diso_start = 0
            for i, diso_residue in enumerate(bool_list):
                if diso_residue != current and diso_residue:        # order to disorder conformation change
                    diso_start = i
                elif diso_residue != current and not diso_residue:    # disorder to order conformation change
                    input.append(embeddings[id][diso_start:i])
                    starts.append(diso_start)
                    stops.append(i)
                    # print(f'{id}: region from {diso_start} to {i}')
                current = diso_residue
            # disorder at final residue?
            if current:
                input.append(embeddings[id][diso_start:])
                starts.append(diso_start)
                stops.append(len(bool_list))
                # print(f'{id}: region from {diso_start} to {stops[-1]}')

            for i, s in enumerate(starts):
                binding = str(labels[id][2][starts[i]:stops[i]])
                binding = re.sub('_', '0', binding)
                binding = list(re.sub(r'B|P|N|O|X|Y|Z|A', '1', binding))
                binding = np.array(binding, dtype=float)
                target.append(binding)

    return input, target


# build the dataset
class BindingDataset(Dataset):
    def __init__(self, embeddings, binding_labels):
        self.inputs = embeddings
        self.labels = binding_labels

    def __len__(self):
        # here the batch size = number of proteins = number of datapoints for the dataloader
        return len(self.labels)

    def number_residues(self):
        return sum([len(protein) for protein in self.labels])

    def __getitem__(self, index):
        # 3-dimensional input must be provided to conv1d, so proteins must be organised in batches
        try:
            return torch.tensor(self.inputs[index]).float(), torch.tensor(self.labels[index], dtype=torch.long)
        except IndexError:
            return None


class CNN(nn.Module):
    def __init__(self, n_layers: int = 5, dropout: float = 0.0, mode: str = 'all'):
        super().__init__()
        self.n_layers = n_layers
        in_c = 1025 if mode == 'all' else 1024
        if self.n_layers == 2:
            # version 0: 2 C layers
            self.conv1 = nn.Conv1d(in_channels=in_c, out_channels=32, kernel_size=5, padding=2)
            # --> out: (32, proteins_length)
            self.dropout = nn.Dropout(p=dropout)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, padding=2)
            # --> out: (1, protein_length)
        elif self.n_layers == 5:
            # version 1: 5 C layers
            self.conv1 = nn.Conv1d(in_channels=in_c, out_channels=512, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, padding=2)
            self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
            self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
            self.conv5 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout)
            # --> out: (1, protein_length)
        else:
            raise ValueError("n_layers must be 2 or 5.")


    def forward(self, input):
        if self.n_layers == 2:
            # version 0: 2 C layers
            x = self.conv1(input.transpose(1, 2).contiguous())
            x = self.dropout(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = x+2
        else:   # 5 layers; check that n_layers is one of 2 or 5 is already done in init()
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
        return x


def train(dataset, model, loss_function, optimizer, device, output):
    avg_train_loss = 0
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    nr_samples = dataset.number_residues()
    for i, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label[None, :].to(device)  # make sure both have same dimensions

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
    test_loss, correct, tp, tp_fn, tp_fp = 0, 0, 0, 0, 0
    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to(device), label[None, :].to(device)
            prediction = model(input)
            test_loss += loss_function(prediction, label.to(torch.float32)).item()
            # apply activation function to prediction to enable classification
            prediction_act = torch.sigmoid(prediction)
            # some arbitrary cutoff. The optimal cutoff will be determined in performance_assessment.py
            prediction_max = prediction_act > 0.1
            # metrics (not final)
            correct += (prediction_max == label).type(torch.float).sum().item()
            tp += (prediction_max == label)[label == 1].type(torch.float).sum().item()
            tp_fn += (label == 1).type(torch.float).sum().item()
            tp_fp += (prediction_max == 1).type(torch.float).sum().item()

            """
            # print(f'val_prediction: {prediction}')
            print(f'val_prediction_activated: {prediction_act}')
            # print(f'val_prediction_max: {prediction_max}')
            print(f'val_labels: {label}')
            print(f'loss: {test_loss}')
            print(f'correct: {correct}')
            print(f'tp: {tp}, tp+fn: {tp_fn}, tp+fp: {tp_fp}')
            """

        test_loss /= size
        correct /= size
        try:
            print(f"\tAccuracy: {(100 * correct):>0.1f}%, Sensitivity: {(100 * (tp / tp_fn)): >0.1f}%, "
                  f"Precision: {(100 * (tp / tp_fp)): >0.1f}%, Avg loss: {test_loss:>8f} \n")
            output.write(f"\tCross-Training set: Accuracy: {(100 * correct):>0.1f}%, "
                         f"Sensitivity: {(100 * (tp / tp_fn)): >0.1f}%, "
                         f"Precision: {(100 * (tp / tp_fp)): >0.1f}%, Avg loss: {test_loss:>8f} \n")
        except ZeroDivisionError:
            print(f"\tAccuracy: {(100 * correct):>0.1f}%, Sensitivity: NA, Precision: NA, "
                  f"Avg loss: {test_loss:>8f} \n")
            output.write(f"\tCross-Training set: Accuracy: {(100 * correct):>0.1f}%, "
                         f"Sensitivity: NA%, Precision: NA, Avg loss: {test_loss:>8f} \n")

    return test_loss


def CNN_trainer(train_embeddings: str, dataset_dir: str, model_name: str = '1_5layers', n_splits: int = 5, oversampling: str = 'binary',
                n_layers: int = 5, dropout: float = 0.0, learning_rate: float = 0.0001, patience: int = 10,
                max_epochs: int = 200, mode: str = 'all'):
    """
    trains the CNN
    :param dataset_dir: directory where the dataset files are stored
    :param train_embeddings: path to the embedding file of the train set datapoints
    :param model_name: name of the model
    :param n_splits: number of Cross-Validation splits
    :param oversampling: oversampling mode; either None or 'binary'
    :param n_layers: number of convolutional layers in the CNN; either 2 or 5
    :param dropout: dropout probability between 0.0 and 1.0
    :param learning_rate: learning rate
    :param max_epochs: max number of epochs before the training stops
    :param patience: early stopping after this number of epochs without improvement
    :param mode: 'all' or 'disorder_only', describes the selection of residues used for training
    """
    # apply cross-validation and oversampling on training dataset
    # CV_and_oversampling.split(n_splits, oversampling)

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
        # for training use all training IDs except for the ones in the current fold.
        # for validation use the training IDs in the current fold

        # read target data y and disorder information
        # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
        val_labels = read_labels(fold, None, dataset_dir)
        train_labels = {}
        for train_fold in range(n_splits):
            if train_fold != fold:
                train_labels.update(read_labels(train_fold, oversampling, dataset_dir))
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
        model = CNN(n_layers, dropout, mode).to(device)
        criterion = nn.BCEWithLogitsLoss()  # loss function for binary problem
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # initialize training parameters for early stopping
        min_val_loss = np.Inf
        epochs_no_improvement = 0
        best_state_dict = None

        output_file = open(f"../results/logs/training_progress_{model_name}_fold_{fold}.txt", "w")

        for epoch in range(max_epochs):
            print(f'{datetime.datetime.now()}\tEpoch {epoch + 1}')
            output_file.write(f'{datetime.datetime.now()}\tEpoch {epoch + 1}\n')
            train(training_dataset, model, criterion, optimizer, device, output_file)
            test_loss = test_performance(validation_dataset, model, criterion, device, output_file)

            # early stopping
            if test_loss < min_val_loss:  # improvement
                min_val_loss = test_loss
                epochs_no_improvement = 0
                best_state_dict = copy.deepcopy(model.state_dict())

            else:  # no improvement
                epochs_no_improvement += 1
                if epochs_no_improvement == patience:
                    break  # -> early stopping

        # save best model of this fold
        torch.save(best_state_dict, f"../results/models/binding_regions_model_{model_name}_fold_{fold}.pth")
        output_file.flush()
        output_file.close()


if __name__ == '__main__':
    CNN_trainer('../dataset/train_set.h5')

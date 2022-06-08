
from src import CV_splits
import numpy as np
import h5py
from Bio import SeqIO
import re
import torch.tensor
from torch.utils.data import Dataset
from torch import nn, optim
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
        # I have to provide 3-dimensional input to conv1d, so proteins must be organised in batches
        try:
            return torch.tensor(self.inputs[index]).float(), torch.tensor(self.labels[index], dtype=torch.long)
        except IndexError:
            return None


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1025, out_channels=32, kernel_size=5, padding=2)
        # --> out: (32, proteins_length)
        # self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, padding=2)
        # --> out: (1, protein_length)

    def forward(self, input):
        x = self.conv1(input.transpose(1, 2).contiguous())
        # x = self.dropout(x)   # dropout makes it worse...
        x = self.relu(x)
        x = self.conv2(x)
        x = x+2
        return x



if __name__ == '__main__':
    oversampling = 'binary'

    # read input embeddings
    embeddings_in = '../dataset/train_set.h5'
    embeddings = dict()
    with h5py.File(embeddings_in, 'r') as f:
        for key, embedding in f.items():
            original_id = embedding.attrs['original_id']
            embeddings[original_id] = np.array(embedding)
    # now {IDs: embeddings} are written in the embeddings dictionary

    def try_cutoffs():
        # iterate over folds
        with open("../results/logs/validation_0_simple_without_dropout.txt", "w") as output_file:
            output_file.write('Fold\tAvg_Loss\tCutoff\tAcc\tPrec\tRec\tTP\tFP\tTN\tFN\n')
            for fold in range(5):
                print("Fold: " + str(fold))
                # for validation use the training IDs in the current fold

                # read target data y and disorder information
                # re-format input information to 3 sequences in a list per protein in dict val/train_labels{}
                val_labels = read_labels(fold, oversampling)

                # create the input and target data exactly how it's fed into the ML model
                # and add the confounding feature of disorder to the embeddings
                this_fold_val_input, this_fold_val_target = get_ML_data(val_labels, embeddings)

                # instantiate the dataset
                validation_dataset = BindingDataset(this_fold_val_input, this_fold_val_target)

                """
                # look at some data:
                for i in range(50, 54):
                   input, label = training_dataset[i]
                   print(f'Embedding input:\n {input}\nPrediction target:\n{label}\n\n')
                """

                def test_performance(dataset, model, loss_function, device, output):
                    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
                    size = dataset.number_residues()
                    # print(size)
                    model.eval()
                    with torch.no_grad():
                        # try out different cutoffs
                        for cutoff in (0.01 * np.arange(0, 41, step=0.5)):   # mult. works around floating point precision issue
                            test_loss, correct, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0
                            for input, label in test_loader:
                                input, label = input.to(device), label[None, :].to(device)
                                prediction = model(input)
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


                            test_loss /= size
                            correct /= size
                            print(
                                f"cutoff {cutoff}\tAccuracy: {(100 * correct):>0.1f}%, Sensitivity: {(100 * (tp / (tp+fn))): >0.1f}%, Precision: {(100 * (tp / (tp+fp))): >0.1f}%, Avg loss: {test_loss:>8f}")
                            output.write('\t'.join([str(fold), str(round(test_loss, 6)), str(cutoff), str(round(100 * correct, 1)),
                                                    str(round(100 * (tp / (tp+fp)), 1)), str(round(100 * (tp / (tp+fn)),1)),
                                                    str(tp), str(fp), str(tn), str(fn)]) + '\n')

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = CNN().to(device)
                model.load_state_dict(
                    torch.load(f"../results/models/binding_regions_model_0_simple_fold_{fold}_without_dropout.pth"))
                # test performance again, should be the same
                criterion = nn.BCEWithLogitsLoss()
                test_performance(validation_dataset, model, criterion, device, output_file)

    def predict(cutoff, fold):
        with open(f"../results/logs/predict_val_0_simple_without_dropout_{fold}_{cutoff}.txt", "w") as output_file:
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
                torch.load(f"../results/models/binding_regions_model_0_simple_fold_{fold}_without_dropout.pth"))
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





    # try_cutoffs()  # expensive!

    # get predictions for chosen cutoff, fold
    cutoff = 0.315
    fold = 0
    predict(cutoff, fold)




from sklearn.model_selection import KFold

if __name__ == '__main__':
    # Input Training Data
    data_dir = '../dataset/'
    with open(data_dir + "train_set_input.txt", 'r') as train_set_labels:
        # Split Training Set for k-fold Cross Validation
        k_fold = KFold(n_splits=5, shuffle=True, random_state=707)
        # generate list of arrays of indices
        label_lines = train_set_labels.readlines()
        subsets = [x[1] for x in k_fold.split(label_lines)]
        # print(subsets)
        # for each subset write annotation to a new file
        for i in enumerate(subsets):
            with open(data_dir + 'folds/' + f"CV_fold_{i[0]}_labels.txt", mode="w") as output_labels:
                entries = [''.join(label_lines[4*x:4*x+4]) for x in subsets[i[0]]]
                for j in entries:
                    output_labels.write(j)
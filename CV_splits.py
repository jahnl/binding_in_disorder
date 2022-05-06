from sklearn.model_selection import KFold

if __name__ == '__main__':
    # Input Training Data
    data_dir = './dataset/'
    with open(data_dir + "train_set_annotation.tsv", 'r') as train_set:
        # Split Training Set for k-fold Cross Validation
        k_fold = KFold(n_splits=5, shuffle=True, random_state=707)
        # generate list of arrays of indices
        train_set_lines = train_set.readlines()[1:]
        subsets = [x[1] for x in k_fold.split(train_set_lines)]
        # print(subsets)
        # for each subset write annotation to a new file
        for i in enumerate(subsets):
            with open(data_dir + 'folds/' + f"CV_fold_{i[0]}.tsv", mode="w") as output:
                output.write("ID\tsequence\tmax.Region.ID\tResidues\tAnnotations\tBindEmbed\n")
                entries = [train_set_lines[x] for x in subsets[i[0]]]
                for j in entries:
                    output.write(j)

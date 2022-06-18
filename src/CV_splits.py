import random
from sklearn.model_selection import KFold
import re


def split(oversampling: str):
    """
    :param oversampling:
    None: no oversampling
    'binary': oversampling for binding(276) vs non-binding(547) proteins --> duplicate all-minus-5 = 98 % of binding proteins --> 1:1
    'binary_residues': binary oversampling on residue-level
        binding(25,051), non-binding(230,173) --> all-binding residues * 9, some (20%) * 10
    ... more options will be implemented (TODO)
    """
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
        # if required: apply oversampling to training set
        for i in enumerate(subsets):
            with open(data_dir + 'folds/' + f"CV_fold_{i[0]}_labels_{oversampling}.txt", mode="w") as output_labels:
                entries = [''.join(label_lines[4 * x:4 * x + 4]) for x in subsets[i[0]]]
                for j in entries:
                    repeat = 1
                    sampled_residues = ['', '', '']
                    try:
                        # is there a binding disordered residue in sequence?
                        if oversampling == 'binary' and re.match(r'.*(P|N|O|X|Y|Z|A).*', j.split('\n')[3]) is not None:
                            chance = (random.randint(1, 100))
                            if chance <= 98:     # only 98% of binding proteins are duplicated
                                repeat = 2
                        # oversample only binding residues, then 'create' new protein with asterisk of only these residues * 9,2
                        elif oversampling == 'binary_residues':
                            # indices of binding residues in this protein
                            indices = [x for x, r in enumerate(j.split('\n')[3]) if re.match(r'(P|N|O|X|Y|Z|A)', r) is not None]
                            line_1 = j.split('\n')[1]
                            line_2 = j.split('\n')[2]
                            line_3 = j.split('\n')[3]
                            sampled_residues[0] = ''.join([line_1[x] for x in indices])
                            sampled_residues[1] = ''.join([line_2[x] for x in indices])
                            sampled_residues[2] = ''.join([line_3[x] for x in indices])
                            if (random.randint(1, 100)) <= 20:
                                repeat = 10
                            else:
                                repeat = 9



                    except IndexError:      # end of file = single line break reached
                        pass
                    if 'residues' not in oversampling:
                        for _ in range(repeat):
                            output_labels.write(j)
                    elif oversampling == 'binary_residues':
                        output_labels.write(j)
                        if sampled_residues != ['', '', '']:
                            output_labels.write(j.split('\n')[0]+'*\t'+((str(indices)[1:-1]+', ')*repeat)+"\n")
                            for _ in range(repeat):
                                output_labels.write(sampled_residues[0])
                            output_labels.write('\n')
                            for _ in range(repeat):
                                output_labels.write(sampled_residues[1])
                            output_labels.write('\n')
                            for _ in range(repeat):
                                output_labels.write(sampled_residues[2])
                            output_labels.write('\n')


if __name__ == '__main__':
    split('binary_residues')

import random
from sklearn.model_selection import KFold
import re

"""
provides split-function that creates Cross-Validation splits and
applies different oversampling techniques for dataset balancing
input: ../dataset/train_set_input.txt
output: ../dataset/folds/CV_fold_[0 - n_splits]_labels[_oversampling].txt
"""


def split(dataset_dir: str, n_splits: int = 5, oversampling: str = 'binary_residues'):
    """
    :param dataset_dir: directory where the dataset files are stored
    :param n_splits: number of Cross-Validation splits
    :param oversampling:
    None: no oversampling
    'binary': oversampling for binding(276) vs non-binding(547) proteins
        --> duplicate all-minus-5 = 98 % of binding proteins --> 1:1
    'binary_residues': binary oversampling on residue-level
        binding(25,051), non-binding(230,173) --> all-binding residues * 9, some (20%) * 10
    'multiclass_residues': multi-class oversampling on residue-level
        non-binding(230,173), protein-binding(19,711), nuc-binding(4,442), other-binding(3,453)
        --> p*11.68, n*51.82, o*66.66
    """
    # Input Training Data
    with open(dataset_dir + "train_set_input.txt", 'r') as train_set_labels:
        # Split Training Set for k-fold Cross Validation
        k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=707)
        # generate list of arrays of indices
        label_lines = train_set_labels.readlines()
        subsets = [x[1] for x in k_fold.split(label_lines)]
        # print(subsets)
        # for each subset write annotation to a new file
        # if required: apply oversampling to training set
        for i in enumerate(subsets):
            with open(f'{dataset_dir}folds/CV_fold_{i[0]}_labels_{oversampling}.txt', mode="w") as output_labels:
                # read out the required set of 4 lines for each protein in subset i
                entries = [''.join(label_lines[4 * x:4 * x + 4]) for x in subsets[i[0]]]
                for j in entries:
                    repeat = 1
                    p_repeat, n_repeat, o_repeat = 0, 0, 0
                    sampled_residues = ['', '', '']
                    p_sampled_residues, n_sampled_residues, o_sampled_residues = ['', '', ''], ['', '', ''], ['', '',
                                                                                                              '']
                    try:
                        # is there a binding disordered residue in sequence?
                        if oversampling == 'binary' and re.match(r'.*(P|N|O|X|Y|Z|A).*', j.split('\n')[3]) is not None:
                            chance = (random.randint(1, 100))
                            if chance <= 98:  # only 98% of binding proteins are duplicated
                                repeat = 2

                        # oversample only binding residues,
                        # then 'create' new protein with asterisk-ID from only these residues * 9.2
                        elif oversampling == 'binary_residues':
                            # indices of binding residues in this protein
                            indices = [x for x, r in enumerate(j.split('\n')[3]) if
                                       re.match(r'(P|N|O|X|Y|Z|A)', r) is not None]
                            line_1 = j.split('\n')[1]
                            line_2 = j.split('\n')[2]
                            line_3 = j.split('\n')[3]
                            sampled_residues[0] = ''.join([line_1[x] for x in indices])
                            sampled_residues[1] = ''.join([line_2[x] for x in indices])
                            sampled_residues[2] = ''.join([line_3[x] for x in indices])
                            if (random.randint(1, 100)) <= 20:
                                repeat = 9  # not 10, bc it's already written 1 time per default
                            else:
                                repeat = 8

                        elif oversampling == 'multiclass_residues':
                            # indices of p-binding residues in this protein
                            p_indices = [x for x, r in enumerate(j.split('\n')[3]) if
                                         re.match(r'(P|X|Y|A)', r) is not None]
                            n_indices = [x for x, r in enumerate(j.split('\n')[3]) if
                                         re.match(r'(N|X|Z|A)', r) is not None]
                            o_indices = [x for x, r in enumerate(j.split('\n')[3]) if
                                         re.match(r'(O|Y|Z|A)', r) is not None]
                            # if there are duplicates in the indices (e.g because of binding class X) delete the index
                            # from the more frequent class(es) (it's always only one (combined) binding class for a
                            # whole protein, so in this case the lists in question should be identical)
                            # --> delete whole majority class list
                            if o_indices == n_indices:
                                n_indices.clear()
                            if o_indices == p_indices:
                                p_indices.clear()
                            if n_indices == p_indices:
                                p_indices.clear()

                            line_1 = j.split('\n')[1]
                            line_2 = j.split('\n')[2]
                            line_3 = j.split('\n')[3]
                            p_sampled_residues[0] = ''.join([line_1[x] for x in p_indices])
                            p_sampled_residues[1] = ''.join([line_2[x] for x in p_indices])
                            p_sampled_residues[2] = ''.join([line_3[x] for x in p_indices])
                            n_sampled_residues[0] = ''.join([line_1[x] for x in n_indices])
                            n_sampled_residues[1] = ''.join([line_2[x] for x in n_indices])
                            n_sampled_residues[2] = ''.join([line_3[x] for x in n_indices])
                            o_sampled_residues[0] = ''.join([line_1[x] for x in o_indices])
                            o_sampled_residues[1] = ''.join([line_2[x] for x in o_indices])
                            o_sampled_residues[2] = ''.join([line_3[x] for x in o_indices])
                            if (random.randint(1, 100)) <= 68:
                                p_repeat = 11
                            else:
                                p_repeat = 10
                            if (random.randint(1, 100)) <= 82:
                                n_repeat = 51
                            else:
                                n_repeat = 50
                            if (random.randint(1, 100)) <= 66:
                                o_repeat = 66
                            else:
                                o_repeat = 65

                    except IndexError:  # end of file = single line break reached
                        pass

                    if 'residues' not in oversampling:
                        for _ in range(repeat):
                            output_labels.write(j)
                    elif oversampling == 'binary_residues':
                        output_labels.write(j)
                        if sampled_residues != ['', '', '']:
                            output_labels.write(
                                j.split('\n')[0] + '*\t' + ((str(indices)[1:-1] + ', ') * repeat) + "\n")
                            for _ in range(repeat):
                                output_labels.write(sampled_residues[0])
                            output_labels.write('\n')
                            for _ in range(repeat):
                                output_labels.write(sampled_residues[1])
                            output_labels.write('\n')
                            for _ in range(repeat):
                                output_labels.write(sampled_residues[2])
                            output_labels.write('\n')
                    elif oversampling == 'multiclass_residues':
                        output_labels.write(j)
                        if p_sampled_residues != sampled_residues or n_sampled_residues != sampled_residues or \
                                o_sampled_residues != sampled_residues:  # sampled_residues is ['','','']
                            indices = p_indices
                            repeat = p_repeat
                            if len(indices) < len(n_indices):
                                indices = n_indices
                                repeat = n_repeat
                            if len(indices) < len(o_indices):
                                indices = o_indices
                                repeat = o_repeat
                            output_labels.write(
                                j.split('\n')[0] + '*\t' + ((str(indices)[1:-1] + ', ') * repeat) + "\n")
                            for _ in range(p_repeat):
                                output_labels.write(p_sampled_residues[0])
                            for _ in range(n_repeat):
                                output_labels.write(n_sampled_residues[0])
                            for _ in range(o_repeat):
                                output_labels.write(o_sampled_residues[0])
                            output_labels.write('\n')
                            for _ in range(p_repeat):
                                output_labels.write(p_sampled_residues[1])
                            for _ in range(n_repeat):
                                output_labels.write(n_sampled_residues[1])
                            for _ in range(o_repeat):
                                output_labels.write(o_sampled_residues[1])
                            output_labels.write('\n')
                            for _ in range(p_repeat):
                                output_labels.write(p_sampled_residues[2])
                            for _ in range(n_repeat):
                                output_labels.write(n_sampled_residues[2])
                            for _ in range(o_repeat):
                                output_labels.write(o_sampled_residues[2])
                            output_labels.write('\n')


if __name__ == '__main__':
    split(5, 'binary_residues')

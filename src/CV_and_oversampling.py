import random
from sklearn.model_selection import KFold
import re

"""
provides split-function that creates Cross-Validation splits and
applies different oversampling techniques for dataset balancing
input: ../dataset/train_set_input.txt
output: ../dataset/folds/CV_fold_[0 - n_splits]_labels[_oversampling].txt
"""


def split(dataset_dir: str, database: str, n_splits: int = 5, oversampling: str = 'binary_residues'):
    """
    :param dataset_dir: directory where the dataset files are stored
    :param n_splits: number of Cross-Validation splits
    :param oversampling: (ratios are different for mobidb dataset!)
    None: no oversampling
    'binary': oversampling for binding(276) vs non-binding(547) proteins
        --> duplicate all-minus-5 = 98 % of binding proteins --> 1:1
    'binary_D': oversampling for binding vs non-binding proteins, but accounting for residue-wise balance
        within disorder
    'binary_residues': binary oversampling on residue-level
        binding(25,051), non-binding(230,173) --> all-binding residues * 9, some (20%) * 10
    'multiclass_residues': multi-class oversampling on residue-level
        non-binding(230,173), protein-binding(19,711), nuc-binding(4,442), other-binding(3,453)
        --> p*11.68, n*51.82, o*66.66
    TODO: and more...
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
        # if required: apply over/under-sampling to training set
        for i in enumerate(subsets):
            with open(f'{dataset_dir}folds/CV_fold_{i[0]}_labels_{oversampling}.txt', mode="w") as output_labels:
                # read out the required set of 4 lines for each protein in subset i
                entries = [''.join(label_lines[4 * x:4 * x + 4]) for x in subsets[i[0]]]
                for j in entries:
                    repeat = 1      # for oversampling of binding residues
                    repeat_neg = 1    # for undersampling of non-binding residues
                    p_repeat, n_repeat, o_repeat = 0, 0, 0
                    sampled_residues = ['', '', ''] # positives in disorder
                    diso_sampled_residues = ['', '', ''] # disorder
                    p_sampled_residues, n_sampled_residues, o_sampled_residues = ['', '', ''], ['', '', ''], ['', '',
                                                                                                              '']
                    try:
                        if oversampling in ['binary', 'binary_D', 'binary_U', 'binary_D_U']:
                            chance = (random.randint(1, 100))
                            disprot_cut = {'binary': 98}    # oversample 98% of binding proteins
                            mobidb_cut = {'binary': 61,     # oversample 61% of binding proteins
                                          'binary_D': 69,   # oversample 69% of binding proteins (based on distr. in disorder)
                                          'binary_U': 38,    # undersample 38% of non-binding proteins
                                          'binary_D_U': 31}  # undersample 31% of non-binding proteins (based on disorder)
                            cut_dir = mobidb_cut if database == 'mobidb' else disprot_cut
                            # is there a binding disordered residue in sequence?
                            if re.match(r'.*(B|P|N|O|X|Y|Z|A).*', j.split('\n')[3]) is not None:
                                if oversampling in ['binary', 'binary_D'] and chance <= cut_dir[oversampling]:
                                    # 98% (61%/69%) of binding proteins are duplicated
                                    repeat = 2
                            # non-binding sequence
                            elif oversampling in ['binary_U', 'binary_D_U'] and chance <= cut_dir[oversampling]:
                                repeat = 0

                        # oversample only binding residues, 'binary_residues(_disorder)'
                        # then 'create' new protein with */$-ID from only these residues * 9.2
                        # (or less when not disprot and binary_residues)
                        elif oversampling is not None and oversampling.startswith('binary_residues'):
                            # indices of binding residues in this protein
                            indices = [x for x, r in enumerate(j.split('\n')[3]) if
                                       re.match(r'(B|P|N|O|X|Y|Z|A)', r) is not None]
                            line_1 = j.split('\n')[1]
                            line_2 = j.split('\n')[2]
                            line_3 = j.split('\n')[3]
                            sampled_residues[0] = ''.join([line_1[x] for x in indices])
                            sampled_residues[1] = ''.join([line_2[x] for x in indices])
                            sampled_residues[2] = ''.join([line_3[x] for x in indices])
                            # indices of disordered residues in this protein
                            diso_indices = [x for x, r in enumerate(j.split('\n')[2]) if r == 'D']
                            diso_sampled_residues[0] = ''.join([line_1[x] for x in diso_indices])
                            diso_sampled_residues[1] = ''.join([line_2[x] for x in diso_indices])
                            diso_sampled_residues[2] = ''.join([line_3[x] for x in diso_indices])
                            if database == 'disprot':
                                if 'D' not in oversampling:     # not disorder-focused balancing
                                    if (random.randint(1, 100)) <= 20:
                                        repeat = 9  # not 10, bc it's already written 1 time per default
                                    else:
                                        repeat = 8
                                else:   # disprot, 'binary_residues_disorder'
                                    # TODO, what's this ratio in disprot dataset?
                                    pass
                            else:   # mobidb
                                if 'D' not in oversampling:  # times 14.2
                                    if (random.randint(1, 100)) <= 20:
                                        repeat = 14  # not 15, bc it's already written 1 time per default
                                    else:
                                        repeat = 13
                                elif 'U' not in oversampling:   # mobidb, 'binary_residues_D', times 1.69,
                                    # negatives outside of disorder: times 0.12
                                    if (random.randint(1, 100)) > 69:
                                        repeat = 0 # not 1, because original positives are always written down once
                                    if (random.randint(1, 100)) > 12:
                                        repeat_neg = 0
                                else:   # mobidb, 'binary_residues_D_U', negatives in disorder: times 0.59,
                                        # negatives outside of disorder: times 0.12
                                    if (random.randint(1, 100)) > 12:
                                            repeat_neg = 0      # --> disorder only
                                            if (random.randint(1, 100)) > (59-12):   # --> positives only
                                                repeat = 0 # special case: this will be used for undersampling
                                    # else: whole protein


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


                    if oversampling is None or 'residues' not in oversampling:
                        for _ in range(repeat):
                            output_labels.write(j)

                    elif oversampling == 'binary_residues':
                        output_labels.write(j)
                        if sampled_residues != ['', '', '']:
                            # in mobidb annotation the fasta id has description -> * would be placed behind the 'id'
                            name = j.split('\n')[0] + '*' if database == 'disprot' else '>*' + j.split('\n')[0][1:]
                            output_labels.write(
                                name + '\t' + ((str(indices)[1:-1] + ', ') * repeat) + "\n")
                            for _ in range(repeat):
                                output_labels.write(sampled_residues[0])
                            output_labels.write('\n')
                            for _ in range(repeat):
                                output_labels.write(sampled_residues[1])
                            output_labels.write('\n')
                            for _ in range(repeat):
                                output_labels.write(sampled_residues[2])
                            output_labels.write('\n')

                    elif oversampling == 'binary_residues_D':
                        if repeat_neg == 1:
                            output_labels.write(j)
                        else: # write down disordered region only
                            name = j.split('\n')[0] + '$' if database == 'disprot' else '>$' + j.split('\n')[0][1:]
                            output_labels.write(
                                name + '\t' + (str(diso_indices)[1:-1] + ', ') + "\n")
                            output_labels.write('\n'.join(diso_sampled_residues) + '\n')
                        if sampled_residues != ['', '', ''] and repeat != 0:
                            # in mobidb annotation the fasta id has description -> * would be placed behind the 'id'
                            name = j.split('\n')[0] + '*' if database == 'disprot' else '>*' + j.split('\n')[0][1:]
                            output_labels.write(
                                name + '\t' + ((str(indices)[1:-1] + ', ') * repeat) + "\n")
                            for _ in range(repeat):
                                output_labels.write(sampled_residues[0])
                            output_labels.write('\n')
                            for _ in range(repeat):
                                output_labels.write(sampled_residues[1])
                            output_labels.write('\n')
                            for _ in range(repeat):
                                output_labels.write(sampled_residues[2])
                            output_labels.write('\n')

                    elif oversampling == 'binary_residues_D_U':
                        if repeat_neg == 1:
                            output_labels.write(j)
                        elif repeat == 1: # write down disordered region only
                            name = j.split('\n')[0] + '$' if database == 'disprot' else '>$' + j.split('\n')[0][1:]
                            output_labels.write(
                                name + '\t' + (str(diso_indices)[1:-1] + ', ') + "\n")
                            output_labels.write('\n'.join(diso_sampled_residues) + '\n')
                        else:   # repeat == 0 --> write down positives only
                            if sampled_residues != ['', '', '']:
                                # in mobidb annotation the fasta id has description -> * would be placed behind the 'id'
                                name = j.split('\n')[0] + '*' if database == 'disprot' else '>*' + j.split('\n')[0][1:]
                                output_labels.write(
                                    name + '\t' + (str(indices)[1:-1] + ', ') + "\n")
                                output_labels.write('\n'.join(sampled_residues) + '\n')

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

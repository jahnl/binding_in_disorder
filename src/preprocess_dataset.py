"""
Preprocessing of the dataset. Includes:
1. statistics about the dataset,
2. an annotation with more information,
3. the input labels for the ML model

input: ../dataset/disprot_annotations.txt, ../dataset/test_set.fasta, ../dataset/train_set.fasta
output: ../dataset/test_set_annotation.tsv, ../dataset/train_set_annotation.tsv, ../dataset/test_set_input.txt,
../dataset/train_set_input.txt
"""

from os.path import exists
from Bio import SeqIO


def sort_dataset(file):
    # parse the fasta-formatted sequences from the input file,
    # sort the dataset (test or train) alphabetically by protein IDs,
    # return a sorted list of protein IDs and AA sequences
    ls = list()
    id, seq = '', ''
    for line in file.readlines():
        if line.startswith('>'):
            if id != '':
                ls.append([id, seq])
                seq = ''
            id = line[1:-1]
        elif line == '':
            ls.append([id, seq])
        else:
            seq += line[:-1]
    return sorted(ls, key=lambda x: x[0])


def ML_input_labels(t_list, t_set, dataset_dir):
    # simpler labelling of the classes: non-binding, protein-binding, nuc-binding and other-binding
    # for each AA sequence annotate the position of disordered and specific binding regions
    # also print out the number of specific residue types
    # is always executed, regardless of 'overwrite' parameter
    ligands = {'non-binding': '_', 'protein': 'P', 'nuc': 'N',
               'lipid': 'O', 'small': 'O', 'metal': 'O', 'ion': 'O', 'carbohydrate': 'O'}
    with open(dataset_dir + t_set + '_set_input.txt', 'w') as out:
        special_case = False
        disorder_str = ''
        ligand_str = ''
        # residue counters
        b_counter, nb_counter, dnb_counter, p_counter, n_counter, o_counter = 0, 0, 0, 0, 0, 0
        for entry in t_list:
            if not special_case:
                out.write('>' + entry[0] + '\n')
                out.write(entry[1] + '\n')
                disorder_str = '-' * len(entry[1])
            for residue in entry[3].split(','):
                disorder_str = disorder_str[:int(residue) - 1] + 'D' + disorder_str[int(residue):]
            binding = list(set(sorted(entry[4].split(','))))
            for i, item in enumerate(binding):
                binding[i] = ligands[item]
            binding = set(binding)
            if len(binding) == 1:
                ligand_class = binding.pop()
            elif len(binding) == 3:
                ligand_class = 'A'
            else:
                if 'P' in binding and 'N' in binding:
                    ligand_class = 'X'
                elif 'P' in binding and 'O' in binding:
                    ligand_class = 'Y'
                else:
                    ligand_class = 'Z'
            if not special_case:
                ligand_str = disorder_str.replace('D', ligand_class)
            else:
                for i, d_residue in enumerate(disorder_str):
                    if d_residue == 'D' and ligand_str[i] == '-':
                        ligand_str = ligand_str[:i] + ligand_class + ligand_str[i + 1:]

            if entry[6] == '>1 regions':
                special_case = True
            else:
                special_case = False
                out.write(disorder_str + '\n')
                out.write(ligand_str + '\n')
                nb_c = ligand_str.count('-') + ligand_str.count('_')
                b_counter += len(ligand_str) - nb_c
                nb_counter += nb_c
                dnb_counter += ligand_str.count('_')
                p_c = ligand_str.count('P') + ligand_str.count('X') + ligand_str.count('Y') + ligand_str.count('A')
                n_c = ligand_str.count('N') + ligand_str.count('X') + ligand_str.count('Z') + ligand_str.count('A')
                o_c = ligand_str.count('O') + ligand_str.count('Y') + ligand_str.count('Z') + ligand_str.count('A')
                p_counter += p_c
                n_counter += n_c
                o_counter += o_c
    print(f'# binding residues: {b_counter}, # non-binding residues: {nb_counter}')
    print(f'# disorder but non-binding residues: {dnb_counter}')
    print(f'# protein-binding residues: {p_counter}, # nuc-binding residues: {n_counter}, '
          f'# other-binding residues: {o_counter}')


def disprot_preprocessing(test_list, train_list, annotations: str, dataset_dir: str, overwrite: bool):
    with open(annotations, 'r') as annotation:
        # sort annotation
        ann_list = list()
        last_id = ''
        for line in annotation.readlines()[1:]:
            tabs = line.rstrip('\n').split('\t')
            tabs.append('')
            # sort residues (some are not in order)
            residues = tabs[2].split(',')
            residues = sorted(residues, key=lambda x: int(x))
            tabs[2] = ','.join(residues)

            if last_id != tabs[0]:
                ann_list.append(tabs)
                last_id = tabs[0]
            else:
                # exception: different regions
                if ann_list[-1][2] != tabs[2]:
                    ann_list[-1][5] = '>1 regions'
                    ann_list.append(tabs)
                    last_id = tabs[0]
                else:  # same region
                    ann_list[-1][1] = tabs[1]
                    ann_list[-1][3] += ',' + tabs[3]
                    ann_list[-1][4] += ',' + tabs[4]
        ann_list = sorted(ann_list, key=lambda x: x[0])

    test_pointer = 0
    train_pointer = 0
    bind_counts_test = {}
    bind_counts_train = {}

    for entry in ann_list:
        ann_id = entry[0]
        try:
            test_id = test_list[test_pointer][0]
        except IndexError:
            test_id = ''
        try:
            train_id = train_list[train_pointer][0]
        except IndexError:
            train_id = ''

        if ann_id != test_id and ann_id != train_id:
            continue
        elif ann_id == test_id:
            if entry[5] == '>1 regions':
                test_list.insert(test_pointer + 1, test_list[test_pointer][:2])
                print(ann_id, 'has more than 1 disordered region! (test set)')
            test_list[test_pointer].extend(entry[1:])
            test_pointer += 1

            # unite list of bindings to sets
            binding = set(sorted(entry[3].split(',')))
            try:
                bind_counts_test[str(binding)] += 1
            except KeyError:
                bind_counts_test[str(binding)] = 1

        elif ann_id == train_id:
            if entry[5] == '>1 regions':
                train_list.insert(train_pointer + 1, train_list[train_pointer][:2])
            train_list[train_pointer].extend(entry[1:])
            train_pointer += 1

            # unite list of bindings to set
            binding = set(entry[3].split(','))
            try:
                bind_counts_train[str(binding)] += 1
            except KeyError:
                bind_counts_train[str(binding)] = 1

        # print('ann_id ' + ann_id + '\n last_id ' + last_id + '\n test_id ' + test_id + '\n test_hit ' + str(test_hit)+
        #      '\n test_pointer ' + str(test_pointer) + '\n')

    test_ann_tsv = dataset_dir + 'test_set_annotation.tsv'
    if not overwrite and exists(test_ann_tsv):
        print(test_ann_tsv + ' already exists, will not be written again.')
    else:
        with open(test_ann_tsv, 'w') as out_test:
            out_test.write('ID\tsequence\tmax.Region.ID\tResidues\tAnnotations\tBindEmbed\t\n')
            for entry in test_list:
                for tab in entry:
                    out_test.write(tab + '\t')
                out_test.write('\n')

    train_ann_tsv = dataset_dir + 'train_set_annotation.tsv'
    if not overwrite and exists(train_ann_tsv):
        print(train_ann_tsv + ' already exists, will not be written again.')
    else:
        with open(train_ann_tsv, 'w') as out_train:
            out_train.write('ID\tsequence\tmax.Region.ID\tResidues\tAnnotations\tBindEmbed\t\n')
            for entry in train_list:
                for tab in entry:
                    out_train.write(tab + '\t')
                out_train.write('\n')

    # write input labels for ML
    ML_input_labels(test_list, 'test', dataset_dir)
    ML_input_labels(train_list, 'train', dataset_dir)

    print('test bind counts: ', bind_counts_test)
    print('train bind counts: ', bind_counts_train)


def mobidb_preprocessing(test_list, train_list, annotations: str, dataset_dir: str, overwrite: bool):
    # rather use a dict for the sequences than a sorted list
    test_dict = {}
    for entry in test_list:
        test_dict[entry[0].split('|')[0]] = [entry[1]]
    train_dict = {}
    for entry in train_list:
        train_dict[entry[0].split('|')[0]] = [entry[1]]

    with open(annotations, 'r') as annotation:
        # add labels to specific set,  ML label generation from different annotation
        # bind_counts_test/train, # binding residues, # non-binding residues, # disorder but non-binding residues
        for record in SeqIO.parse(annotation, 'fasta'):
            id = record.id.split('|')[0]
            if id in train_dict and 'sequence' not in record.id:
                train_dict[id].append(str(record.seq))
            elif id in test_dict and 'sequence' not in record.id:
                test_dict[id].append(str(record.seq))
            # else: has been excluded from development set
    for set in [test_dict, train_dict]:
        name = 'test' if set == test_dict else 'train'
        with open(dataset_dir + name + '_set_input.txt', 'w') as out:
            for entry_id in set.keys():
                set[entry_id][1] = set[entry_id][1].replace('0', '-').replace('1', 'D')     # adapt annotation
                if len(set[entry_id]) < 3:
                    set[entry_id].append('-'*len(set[entry_id][0]))  # add labels for 'no LIP'
                else:
                    set[entry_id][2] = set[entry_id][2].replace('1', 'B').replace('0', '-')     # adapt annotation
                for i, residue in enumerate(set[entry_id][1]):
                    if residue == 'D' and set[entry_id][2][i] == '-':
                        set[entry_id][2] = set[entry_id][2][:i] + '_' + set[entry_id][2][i+1:]
                    elif residue == '-' and set[entry_id][2][i] == 'B':
                        # exclude binding regions outside of disordered regions
                        set[entry_id][2] = set[entry_id][2][:i] + '-' + set[entry_id][2][i + 1:]
                # write ML input file
                out.write('>' + entry_id + '\n' + set[entry_id][0] + '\n' + set[entry_id][1] + '\n' + set[entry_id][2]
                          + '\n')


def preprocess(test_set_fasta: str, train_set_fasta: str, annotations: str, database: str, dataset_dir: str,
               overwrite: bool):
    # match the annotation with the data actually used, different computation depending on database
    # write new, more useful annotation, if database==disprot
    # print statistics
    # call ML input label creation

    # sort the train and test set to enable fast access for a later point in the workflow
    with open(test_set_fasta, 'r') as test_set:
        test_list = sort_dataset(test_set)
    with open(train_set_fasta, 'r') as train_set:
        train_list = sort_dataset(train_set)

    if database == 'disprot':
        disprot_preprocessing(test_list, train_list, annotations, dataset_dir, overwrite)
    elif database == 'mobidb':  # based on mobidb's fasta-like output format, with merge and disorder annotation
        mobidb_preprocessing(test_list, train_list, annotations, dataset_dir, overwrite)

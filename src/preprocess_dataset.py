"""
Preprocessing of the dataset. Includes:
1. statistics about the dataset,
2. an annotation with more information,
3. the input labels for the ML model

input: ../dataset/disprot_annotations.txt, ../dataset/test_set.fasta, ../dataset/train_set.fasta
output: ../dataset/test_set_annotation.tsv, ../dataset/train_set_annotation.tsv, ../dataset/test_set_input.txt,
../dataset/train_set_input.txt
"""
from math import ceil
import random
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
        else:
            seq += line[:-1]
    ls.append([id, seq])    # last entry
    # print("sorting: ID in there? ", ls[-1])
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


def mobidb_preprocessing(test_list, train_list, annotations: list, disorder_param: str, dataset_dir: str, overwrite: bool):
    random.seed(303)
    # rather use a dict for the sequences than a sorted list
    test_dict = {}
    for entry in test_list:
        test_dict[entry[0]] = [entry[1]]    # test_dict[id[0]] = sequence
    train_dict = {}
    if train_list is not None:
        for entry in train_list:
            train_dict[entry[0]] = [entry[1]]

    if len(annotations) == 1:       # MobiDB annotation in FASTA format
        with open(annotations[0], 'r') as annotation:
            # add labels to specific set,  ML label generation from different annotation
            # (bind_counts_test/train (per protein...))
            # # binding residues, # non-binding residues, # disorder but non-binding residues
            ann_no = 0
            long_id = ''
            lip_seq = ''
            diso_seq = ''
            seq_length = 0
            for record in SeqIO.parse(annotation, 'fasta'):
                # parse sets of parameters, in variable order
                # binding: always curated-lip-merge;
                # disorder: depending on disorder_param, default: curated-disorder-merge

                # first sequence, then 1 to n different annotations, including the ones that will be used
                if '|sequence' in record.description:   # new entry
                    # save last entry, if in development set and only if sequence length == annotation length
                    if len(diso_seq) == seq_length:
                        if ann_no == 1:
                            test_dict[long_id].append(diso_seq)
                            if lip_seq != '':
                                test_dict[long_id].append(lip_seq)
                        elif ann_no == -1:
                            train_dict[long_id].append(diso_seq)
                            if lip_seq != '':
                                train_dict[long_id].append(lip_seq)
                    elif ann_no != 0:
                        print(f'Excluding {long_id} from development set for having inconsistent annotation lengths.')
                        if ann_no == 1:
                            test_dict.pop(long_id)
                        else:
                            train_dict.pop(long_id)
                    lip_seq = ''
                    # start new entry
                    long_id = record.description
                    seq_length = len(record.seq)
                    if long_id in test_dict:
                        ann_no = 1
                    elif long_id in train_dict:
                        ann_no = -1
                    else:  # has been excluded from development set
                        ann_no = 0

                elif ann_no != 0:   # annotation in dev. set
                    if disorder_param in record.id:
                        diso_seq = str(record.seq)
                    elif 'curated-lip-merge' in record.id:
                        lip_seq = str(record.seq)

            # save last entry, if in development set
            if ann_no == 1:
                test_dict[long_id].append(diso_seq)
                if lip_seq != '':
                    test_dict[long_id].append(lip_seq)
            elif ann_no == -1:
                test_dict[long_id].append(diso_seq)
                if lip_seq != '':
                    test_dict[long_id].append(lip_seq)

    else:       # SETH prediction in CSV and TXT format, test set only
        annotations = sorted(annotations, key=lambda x: x[-3])  # csv before txt
        with open(annotations[1]) as txt_id:
            with open(annotations[0]) as csv:
                csv_pr = csv.readlines()
                for i, id in enumerate(txt_id.readlines()):
                    disorder_pr = csv_pr[i].split(', ')
                    # parse CheZOD scores: score >= 8 --> Order, score < 8 --> Disorder
                    disorder_str = ''
                    for score in disorder_pr:
                        if float(score) >= 8:
                            disorder_str += '-'
                        else:
                            disorder_str += 'D'
                    try:
                        test_dict[id[1:-1]].extend([disorder_str, '/'])    # no binding annotation!
                    except KeyError:    # last entry without \n at the end...
                        test_dict[id[1:]].extend([disorder_str, '/'])

    dicts = [test_dict, train_dict] if len(annotations) == 1 and train_list is not None else [test_dict]
    for set in dicts:
        per_protein_counts = {'length': [], 'n_disordered': [], 'n_structured': [],
                              'n_D_binding': [], 'n_D_nonbinding': [],
                              'binding_positioning_distr': [0, 0, 0, 0, 0],
                              'D_region_length': []}
        per_protein_score = []
        per_protein_score_2 = []
        # last entry: distribution of position of binding residues within the disordered regions, in 5ths; a.k.a. positional bias

        bind_count = 0
        nbind_count = 0
        diso_nbind_count = 0
        positive_proteins = 0
        negative_proteins = 0
        pos_prot_res = 0
        neg_prot_res = 0
        name = 'test' if set == test_dict else 'train'
        seth = '_seth' if len(annotations) == 2 else ''
        with open(f"{dataset_dir}{name}_set{seth}_input.txt", 'w') as out:
            for entry_id in set.keys():
                set[entry_id][1] = set[entry_id][1].replace('0', '-').replace('1', 'D')     # adapt annotation
                if len(set[entry_id]) < 3:
                    set[entry_id].append('-'*len(set[entry_id][0]))  # add labels for 'no LIP'
                else:
                    set[entry_id][2] = set[entry_id][2].replace('1', 'B').replace('0', '-')     # adapt annotation

                if seth == '':
                    disordered_regions = []
                    last_in_D = False
                    for i, residue in enumerate(set[entry_id][1]):
                        if residue == 'D':
                            # report disordered region
                            if not last_in_D:
                                disordered_regions.append("")
                                last_in_D = True
                            disordered_regions[-1] += set[entry_id][2][i]
                            # change annotation of non-binding in disorder
                            if set[entry_id][2][i] == '-':
                                set[entry_id][2] = set[entry_id][2][:i] + '_' + set[entry_id][2][i+1:]

                        elif residue == '-':
                            # change to structured region
                            last_in_D = False
                            # exclude binding regions outside of disordered regions
                            if set[entry_id][2][i] == 'B':
                                set[entry_id][2] = set[entry_id][2][:i] + '-' + set[entry_id][2][i + 1:]

                    # statistics
                    b_c = set[entry_id][2].count('B')
                    length = len(set[entry_id][2])
                    bind_count += b_c
                    nbind_count += length - b_c
                    diso_nb_c = set[entry_id][2].count('_')
                    diso_nbind_count += diso_nb_c
                    if b_c > 0:
                        positive_proteins += 1
                        pos_prot_res += length
                    else:
                        negative_proteins += 1
                        neg_prot_res += length

                    # counts for distribution:
                    per_protein_counts['length'].append(length)
                    per_protein_counts['n_disordered'].append(b_c + diso_nb_c)
                    per_protein_counts['n_structured'].append(length - (b_c + diso_nb_c))
                    per_protein_counts['n_D_binding'].append(b_c)
                    per_protein_counts['n_D_nonbinding'].append(diso_nb_c)
                    for region in disordered_regions:
                        per_protein_counts['D_region_length'].append(len(region))
                        chunk_size = ceil(len(region) / 5)
                        double_residues = (5 - (len(region) - (chunk_size - 1) * 5)) % 5
                        insert_positions = random.sample(population=range(len(region)), k=double_residues)
                        insert_positions.sort(reverse=True)
                        for p in insert_positions:
                            region = region[:p] + region[p]*2 + region[p+1:]
                        chunks = [region[i:i + chunk_size] for i in range(0, len(region), chunk_size)]
                        for i, c in enumerate(chunks):
                            per_protein_counts['binding_positioning_distr'][i] += c.count('B')

                    score = round(length + (500 * (diso_nb_c/length)), 2)
                    per_protein_score.append((entry_id.split('|')[0], score))
                    per_protein_score_2.append((entry_id.split('|')[0], round(length, 2)))

                # write ML input file
                out.write('>' + entry_id + '\n' + set[entry_id][0] + '\n' + set[entry_id][1] + '\n' + set[entry_id][2]
                          + '\n')
        if seth == '':
            print(f'{name} set:\nbinding residues: {bind_count}\nnon-binding residues: {nbind_count}\n'
                  f'non-binding residues in disorder: {diso_nbind_count}\npositive proteins: {positive_proteins} with '
                  f'{pos_prot_res} residues\nnegative proteins: {negative_proteins} with {neg_prot_res} residues\n')
            # print(per_protein_counts)

            with open(f"{dataset_dir}{name}_set{seth}_stats.txt", 'w') as out:
                for key in per_protein_counts.keys():
                    out.write(key + "\n" + str(per_protein_counts[key]) + "\n")
            with open(f"{dataset_dir}{name}_set{seth}_score_distribution.tsv", 'w') as out:
                out.write("protein\tscore\n")
                for element in per_protein_score:
                    out.write(element[0] + "\t" + str(element[1]) + "\n")

        """
        # statistics for CV-folds (only possible if they already exist)
        for fold in range(5):
            per_protein_counts = {'length': [], 'n_disordered': [], 'n_structured': [],
                                  'n_D_binding': [], 'n_D_nonbinding': [],
                                  'binding_positioning_distr': [0, 0, 0, 0, 0],
                                  'D_region_length': []}
            per_protein_score = []

            bind_count = 0
            nbind_count = 0
            diso_nbind_count = 0
            positive_proteins = 0
            negative_proteins = 0
            pos_prot_res = 0
            neg_prot_res = 0
            with open(dataset_dir + 'folds/CV_fold_' + str(fold) +'_labels_None.txt', 'r') as fold_in:
                triple_pos = -1
                for line in fold_in.readlines():
                    triple_pos = (triple_pos + 1) % 4
                    if triple_pos == 3:     # look only at the line with binding annotation, one protein
                        disordered_regions = []
                        last_in_D = False
                        for residue in line:
                            if residue in ['_', 'B']:
                                # report disordered region
                                if not last_in_D:
                                    disordered_regions.append("")
                                    last_in_D = True
                                disordered_regions[-1] += residue

                            elif residue == '-':
                                # change to structured region
                                last_in_D = False

                        # statistics
                        b_c = line.count('B')
                        length = len(line) - 1
                        bind_count += b_c
                        nbind_count += length - b_c
                        diso_nb_c = line.count('_')
                        diso_nbind_count += diso_nb_c
                        if b_c > 0:
                            positive_proteins += 1
                            pos_prot_res += length
                        else:
                            negative_proteins += 1
                            neg_prot_res += length

                        # counts for distribution:
                        per_protein_counts['length'].append(length)
                        per_protein_counts['n_disordered'].append(b_c + diso_nb_c)
                        per_protein_counts['n_structured'].append(length - (b_c + diso_nb_c))
                        per_protein_counts['n_D_binding'].append(b_c)
                        per_protein_counts['n_D_nonbinding'].append(diso_nb_c)
                        for region in disordered_regions:
                            per_protein_counts['D_region_length'].append(len(region))
                            chunk_size = ceil(len(region) / 5)
                            double_residues = (5 - (len(region) - (chunk_size - 1) * 5)) % 5
                            insert_positions = random.sample(population=range(len(region)), k=double_residues)
                            insert_positions.sort(reverse=True)
                            for p in insert_positions:
                                region = region[:p] + region[p] * 2 + region[p + 1:]
                            chunks = [region[i:i + chunk_size] for i in range(0, len(region), chunk_size)]
                            for i, c in enumerate(chunks):
                                per_protein_counts['binding_positioning_distr'][i] += c.count('B')

                        score = round(length + (500 * (diso_nb_c / length)), 2)
                        per_protein_score.append((entry_id.split('|')[0], score))
                        #per_protein_score_2.append((entry_id.split('|')[0], round(length, 2)))

            with open(dataset_dir + 'val_fold_' + str(fold) + '_stats.txt', 'w') as out:
                for key in per_protein_counts.keys():
                    out.write(key + "\n" + str(per_protein_counts[key]) + "\n")
            with open(dataset_dir + 'val_fold_' + str(fold) + '_score_distribution.tsv', 'w') as out:
                out.write("protein\tscore\n")
                for element in per_protein_score:
                    out.write(element[0] + "\t" + str(element[1]) + "\n")
        """


def preprocess(test_set_fasta: str, train_set_fasta: str, annotations: list, disorder_param: str,
               database: str, dataset_dir: str, overwrite: bool):
    # match the annotation with the data actually used, different computation depending on database
    # write new, more useful annotation, if database==disprot
    # print statistics
    # call ML input label creation

    # sort the train and test set to enable fast access for a later point in the workflow
    with open(test_set_fasta, 'r') as test_set:
        test_list = sort_dataset(test_set)
    if train_set_fasta != '':
        with open(train_set_fasta, 'r') as train_set:
            train_list = sort_dataset(train_set)
    else:
        train_list = None

    if database == 'disprot':
        disprot_preprocessing(test_list, train_list, annotations[0], dataset_dir, overwrite)
    elif database == 'mobidb':  # based on mobidb's fasta-like output format, with merge and disorder annotation
        mobidb_preprocessing(test_list, train_list, annotations, disorder_param, dataset_dir, overwrite)

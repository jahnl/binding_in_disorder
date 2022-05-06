
"""
input: ./dataset/disprot_annotations.txt, ./dataset/test_set.fasta, ./dataset/train_set.fasta
output: statistics about the dataset
"""

def sort_dataset(file):
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




if __name__ == '__main__':
    # match the annotation with the data actually used
    # sort the train and test set to enable fast access later
    with open('../dataset/test_set.fasta', 'r') as test_set:
        test_list = sort_dataset(test_set)
    with open('../dataset/train_set.fasta', 'r') as train_set:
        train_list = sort_dataset(train_set)

    with open('../dataset/disprot_annotations.txt', 'r') as annotation:
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
                else:   # same region
                    ann_list[-1][1] = tabs[1]
                    ann_list[-1][3] += ','+tabs[3]
                    ann_list[-1][4] += ','+tabs[4]
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
            binding = set(entry[3].split(','))
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

        # print('ann_id ' + ann_id + '\n last_id ' + last_id + '\n test_id ' + test_id + '\n test_hit ' + str(test_hit) +
        #      '\n test_pointer ' + str(test_pointer) + '\n')

    with open('../dataset/test_set_annotation.tsv', 'w') as out_test:
        out_test.write('ID\tsequence\tmax.Region.ID\tResidues\tAnnotations\tBindEmbed\t\n')
        for entry in test_list:
            for tab in entry:
                out_test.write(tab + '\t')
            out_test.write('\n')
    with open('../dataset/train_set_annotation.tsv', 'w') as out_train:
        out_train.write('ID\tsequence\tmax.Region.ID\tResidues\tAnnotations\tBindEmbed\t\n')
        for entry in train_list:
            for tab in entry:
                out_train.write(tab + '\t')
            out_train.write('\n')

    print('test bind counts: ', bind_counts_test)
    print('train bind counts: ', bind_counts_train)








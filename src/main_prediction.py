import configparser
from os.path import exists
import shutil
from datetime import datetime
import re

import src.preprocess_dataset
import src.investigate_model
import src.sampling_datapoints


# Method to read config file
def read_config():
    config = configparser.ConfigParser()
    config.read('../config_prediction.ini')
    return config


def get_optimal_fold(model_name):
    best_fold = {"mobidb_CNN_0": 2,
                 "mobidb_CNN_1": 1,
                 "mobidb_CNN_2": 1,
                 "mobidb_FNN_0": 1,
                 "mobidb_FNN_1": 3,
                 "mobidb_FNN_2": 2,
                 "mobidb_FNN_3": 2,
                 "mobidb_FNN_4": 2,
                 "mobidb_FNN_5": 2,
                 "mobidb_D_CNN_0": 2,
                 "mobidb_D_CNN_0_lr0005": 2,
                 "mobidb_D_CNN_0_lr001": 2,
                 "mobidb_D_CNN_0_d2": 4,
                 "mobidb_D_CNN_0_d3": 3,
                 "mobidb_D_CNN_0_k3": 2,
                 "mobidb_D_CNN_0_k7": 2,
                 "mobidb_D_CNN_0_l8": 3,
                 "mobidb_D_CNN_1": 2,
                 "mobidb_D_CNN_2": 2,
                 "mobidb_D_FNN_0": 1,
                 "mobidb_D_FNN_1": 2,
                 "mobidb_D_FNN_2": 4,
                 "mobidb_D_FNN_3": 2,
                 "mobidb_D_FNN_4": 0,
                 "mobidb_2_CNN_0": 2,
                 "mobidb_2_CNN_1": 1,   # 1: best CNN trained on all residues, CNN_all
                 "mobidb_2_CNN_2": 2,
                 "mobidb_2_FNN_0": 4,
                 "mobidb_2_FNN_1": 1,
                 "mobidb_2_FNN_2": 0,
                 "mobidb_2_FNN_3": 0,
                 "mobidb_2_FNN_4": 2,
                 "mobidb_2_FNN_5": 4,  # 3: final model, FNN_all
                 "mobidb_2_D_CNN_0": 2,
                 "mobidb_2_D_CNN_1": 0,
                 "mobidb_2_D_CNN_2": 3,  # 2: best CNN, CNN_disorder
                 "mobidb_2_D_FNN_0": 2,
                 "mobidb_2_D_FNN_1": 0,
                 "mobidb_2_D_FNN_2": 0,
                 "mobidb_2_D_FNN_3": 4,
                 "mobidb_2_D_FNN_4": 1,  # 4: best FNN trained on disorder only, FNN_disorder
                 "random_binary": 2,
                 "random_D_only": 2,
                 "AAindex_baseline": 4,  # based on mobidb_CNN_0
                 "AAindex_D_baseline": 1,  # based on mobidb_D_CNN_0
                 "AAindex_baseline_2": 3,  # based on mobidb_2_CNN_1
                 "AAindex_D_baseline_2": 3  # based on mobidb_2_D_CNN_2
                 }
    return best_fold[model_name]


def get_optimal_cutoff(model_name, fold):
    # cutoffs are different for each fold and variant
    cutoffs = {"mobidb_CNN_0": [0.35, 0.3, 0.3, 0.15, 0.4],
               "mobidb_CNN_1": [0.1, 0.4, 0.35, 0.5, 0.55],
               "mobidb_CNN_2": [0.15, 0.3, 0.35, 0.55, 0.45],
               "mobidb_FNN_0": [0.35, 0.35, 0.4, 0.45, 0.35],
               "mobidb_FNN_1": [0.9, 0.9, 0.9, 0.9, 0.85],
               "mobidb_FNN_2": [0.55, 0.55, 0.6, 0.6, 0.65],
               "mobidb_FNN_3": [0.4, 0.5, 0.45, 0.2, 0.4],
               "mobidb_FNN_4": [0.4, 0.4, 0.45, 0.45, 0.5],
               "mobidb_FNN_5": [0.65, 0.55, 0.45, 0.5, 0.55],
               "mobidb_D_CNN_0": [0.4, 0.35, 0.45, 0.35, 0.35],
               "mobidb_D_CNN_0_lr0005": [0.45, 0.45, 0.5, 0.45, 0.2],
               "mobidb_D_CNN_0_lr001": [0.5, 0.5, 0.5, 0.55, 0.25],
               "mobidb_D_CNN_0_d2": [0.55, 0.55, 0.55, 0.55, 0.5],
               "mobidb_D_CNN_0_d3": [0.15, 0.5, 0.25, 0.55, 0.3],
               "mobidb_D_CNN_0_k3": [0.45, 0.4, 0.5, 0.5, 0.3],
               "mobidb_D_CNN_0_k7": [0.2, 0.2, 0.45, 0.35, 0.5],
               "mobidb_D_CNN_0_l8": [0.25, 0.25, 0.25, 0.5, 0.25],
               "mobidb_D_CNN_1": [0.25, 0.2, 0.25, 0.25, 0.45],
               "mobidb_D_CNN_2": [0.35, 0.25, 0.4, 0.25, 0.25],
               "mobidb_D_FNN_0": [0.4, 0.35, 0.4, 0.35, 0.4],
               "mobidb_D_FNN_1": [0.5, 0.5, 0.55, 0.5, 0.5],
               "mobidb_D_FNN_2": [0.4, 0.4, 0.4, 0.45, 0.4],
               "mobidb_D_FNN_3": [0.45, 0.45, 0.4, 0.45, 0.5],
               "mobidb_D_FNN_4": [0.6, 0.55, 0.55, 0.5, 0.5],
               "mobidb_2_CNN_0": [0.15, 0.55, 0.5, 0.5, 0.15],
               "mobidb_2_CNN_1": [0.4, 0.35, 0.6, 0.55, 0.15],
               "mobidb_2_CNN_2": [0.4, 0.45, 0.55, 0.45, 0.15],
               "mobidb_2_FNN_0": [0.35, 0.45, 0.45, 0.4, 0.35],
               "mobidb_2_FNN_1": [0.85, 0.85, 0.85, 0.85, 0.85],
               "mobidb_2_FNN_2": [0.45, 0.5, 0.5, 0.5, 0.35],
               "mobidb_2_FNN_3": [0.4, 0.45, 0.4, 0.45, 0.25],
               "mobidb_2_FNN_4": [0.15, 0.5, 0.45, 0.45, 0.2],
               "mobidb_2_FNN_5": [0.55, 0.4, 0.5, 0.35, 0.3],
               "mobidb_2_D_CNN_0": [0.4, 0.2, 0.4, 0.45, 0.2],
               "mobidb_2_D_CNN_1": [0.35, 0.25, 0.45, 0.2, 0.25],
               "mobidb_2_D_CNN_2": [0.4, 0.35, 0.3, 0.45, 0.3],
               "mobidb_2_D_FNN_0": [0.4, 0.4, 0.4, 0.4, 0.4],
               "mobidb_2_D_FNN_1": [0.45, 0.5, 0.5, 0.45, 0.5],
               "mobidb_2_D_FNN_2": [0.45, 0.5, 0.4, 0.5, 0.3],
               "mobidb_2_D_FNN_3": [0.45, 0.45, 0.45, 0.45, 0.4],
               "mobidb_2_D_FNN_4": [0.5, 0.45, 0.5, 0.45, 0.5],
               "random_binary": [0.94, 0.94, 0.94, 0.94, 0.94],
               "random_D_only": [0.63, 0.63, 0.63, 0.63, 0.63],
               "AAindex_baseline": [0.1, 0.1, 0.15, 0.1, 0.1],
               "AAindex_D_baseline": [0.25, 0.2, 0.25, 0.25, 0.25],
               "AAindex_baseline_2": [0.05, 0.2, 0.05, 0.15, 0.1],
               "AAindex_D_baseline_2": [0.3, 0.3, 0.25, 0.25, 0.25]
               }
    return cutoffs[model_name][fold]


def check_config_items(model_name, config, annotations):
    # check parameters for correctness
    if config['workflow']['overwrite'] not in ['True', 'False']:
        raise ValueError("Config item 'overwrite' must be 'True' or 'False'.")

    bad_c = '/\\:*?"<>|'
    for c in bad_c:
        if c in model_name:
            raise ValueError("The following characters are not allowed in config item 'model_name': "
                             '/\\:*?"<>|"')

    for a in annotations:
        if not exists(a):
            raise ValueError(f"Config item 'annotations': {a} is no existing file.")
    if len(annotations) == 1 and not annotations[0].lower().endswith('.fasta'):
        raise ValueError(f"Config item 'annotations': file must be a MobiDB annotation in FASTA format.")
    elif len(annotations) == 2:
        formats = sorted([x[-4:] for x in annotations])
        if formats != ['.csv', '.txt']:
            raise ValueError(f"Config item 'annotations': files must be a SETH disorder prediction in CSV format "
                             f"and the respective sequence-ID list in TXT format.")
    elif len(annotations) > 2:
        raise ValueError("Config item 'annotations': too many items. Must be either FASTA-file with MobiDB "
                         "annotation, or disorder prediction (CSV) and ID (TXT) files generated by SETH")

    if not exists(config['input_files']['test_set_fasta']):
        raise ValueError(f"Config item 'test_set_fasta': {config['input_files']['test_set_fasta']} is no existing "
                         f"file.")

    if config['parameters']['fold'] != '':
        if not config['parameters']['fold'].isnumeric() or not 0 <= int(config['parameters']['fold']) < 5:
            raise ValueError("Config item 'fold' must be blank, or an integer between 0 and 4.")

    pattern = re.compile('0\.[0-9]+\Z')
    if config['parameters']['cutoff'] != '' and pattern.match(config['parameters']['cutoff']) is None:
        raise ValueError("Config item 'cutoff' must be blank, or a single float, between 0.0 and 1.0.")

    if not exists(config['input_files']['test_set_embeddings']) and not config['input_files'][
                                                                            'test_set_embeddings'] == '':
        raise ValueError(f"Config item 'test_set_embeddings': {config['input_files']['test_set_embeddings']} "
                         f"is no existing file.")
    if config['input_files']['test_set_embeddings'] != '' and 'AAindex' in model_name:
        raise UserWarning("If you want to use AAindex input instead of ProtT5 embeddings, "
                          "leave the config item 'test_set_embeddings' blank.")


if __name__ == '__main__':
    start = datetime.now()
    dt = str(start)[:19].replace(':', '-')
    config = read_config()

    annotations = config['input_files']['annotations'].replace(' ', '').split(',')

    model_alias = {'CNN_all': 'mobidb_2_CNN_1',
                   'CNN_disorder': 'mobidb_2_D_CNN_2',
                   'FNN_all': 'mobidb_2_FNN_5',
                   'FNN_disorder': 'mobidb_2_D_FNN_4',
                   'AAindex_disorder': 'AAindex_D_baseline_2'}
    model_name = config['parameters']['model_name']
    if model_name in model_alias:
        model_name = model_alias[model_name]
    elif model_name == '':
        model_name = 'mobidb_2_FNN_5'

    # parse potential empty strings to optimal value stored in dictionary
    fold = config['parameters']['fold']
    fold = get_optimal_fold(model_name) if fold == '' else int(fold)
    cutoff = config['parameters']['cutoff']
    cutoff = get_optimal_cutoff(model_name, fold) if cutoff == '' else float(cutoff)
    residues = 'disorder_only' if '_D_' in model_name else 'all'

    # parse bools
    wf_overwrite = config['workflow']['overwrite'] != 'False'

    # check for parameter correctness
    check_config_items(model_name, config, annotations)

    if not wf_overwrite and \
            ((len(annotations) == 1 and exists(config['input_files']['dataset_directory'] + 'test_set_input.txt')) or
             (len(annotations) == 2 and exists(config['input_files']['dataset_directory'] + 'test_set_seth_input.txt'))):
        print('skipping preprocessing, necessary files are already present')
    else:
        print('preprocessing dataset...')
        src.preprocess_dataset.preprocess(test_set_fasta=config['input_files']['test_set_fasta'],
                                          train_set_fasta='',
                                          annotations=annotations,
                                          database='mobidb',
                                          dataset_dir=config['input_files']['dataset_directory'],
                                          overwrite=wf_overwrite)

    # in case of AAindex usage, create AAindex representation, if not already there
    if config['input_files']['test_set_embeddings'] == '':
        if exists(config['input_files']['dataset_directory'] + 'AAindex_representation.npy') and not wf_overwrite:
            print('skipping creation of AAindex representation, necessary files are already present')
        else:
            print('creating AAindex representation...')
            src.sampling_datapoints.sample_datapoints(train_embeddings='',
                                                      dataset_dir=config['input_files']['dataset_directory'],
                                                      database='mobidb',
                                                      oversampling='',
                                                      mode=residues,
                                                      n_splits=0)   # 0, because no representation must be created for any train set split


    if not wf_overwrite:
        all_val_files_present = False
        if exists(f"../results/logs/predict_val_{model_name}_{cutoff}_test.txt"):
            all_val_files_present = True
    if not wf_overwrite and all_val_files_present:
        print('skipping prediction, all output files are already present')
    else:
        print('predicting...')
        src.investigate_model.predict(train_embeddings='',
                                      test_embeddings=config['input_files']['test_set_embeddings'],
                                      dataset_dir=config["input_files"]["dataset_directory"],
                                      annotations=annotations,
                                      model_name=model_name,
                                      consensus_models='',
                                      consensus_folds='',
                                      fold=fold,
                                      cutoff=cutoff,
                                      mode=residues,
                                      architecture='FNN' if 'FNN' in model_name else 'CNN',     # CNN also for AAindex models!
                                      multilabel=False,
                                      n_layers=8 if model_name.endswith('_l8') else 5,
                                      kernel_size=int(model_name[-1]) if model_name[-3:-2] == '_k' else 5,
                                      dropout=int(model_name[-1])/10 if model_name[-3:-2] == '_d' else 0.0,
                                      batch_size=512,
                                      test=True,
                                      post_processing=False
                                      )

    # copy config file for documentation of parameters
    shutil.copyfile(src='../config_prediction.ini', dst=f"../results/logs/config_prediction_{model_name}_{dt}.ini")
    print("done.\nruntime: ", datetime.now() - start)

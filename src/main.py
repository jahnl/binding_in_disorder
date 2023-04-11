import configparser
from os.path import exists
import shutil
from datetime import datetime
import string
import re

import src.preprocess_dataset
import src.CV_and_oversampling
import src.sampling_datapoints
import src.trainer_0_CNN
import src.trainer_1_FNN
import src.trainer_2_multilabel_FNN
import src.investigate_model


# Method to read config file
def read_config():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    return config


def check_config_items(step, config, consensus_models, folds, annotations):
    # check parameters for correctness
    # for all steps
    if config['workflow']['overwrite'] not in ['True', 'False']:
        raise ValueError("Config item 'overwrite' must be 'True' or 'False'.")

    # for subsets of steps
    if step in [1, 2]:      # TODO: for steps 1,2, and 3!
        if not config['parameters']['database'] in ['disprot', 'mobidb']:
            raise ValueError("Config item 'database' must be 'disprot' or 'mobidb'.")
    elif step in [3, 4, 5, 7]:
        if not exists(config['input_files']['train_set_embeddings']) and not config['input_files']['train_set_embeddings'] == '':
            raise ValueError(f"Config item 'train_set_embeddings': {config['input_files']['train_set_embeddings']} "
                             f"is no existing file.")
        if step != 3:
            if not config['parameters']['residues'] in ['all', 'disorder_only']:
                raise ValueError("Config item 'residues' must be 'all' or 'disorder_only'.")

            bad_c = '/\\:*?"<>|'
            for c in bad_c:
                if c in config['parameters']['model_name']:
                    raise ValueError("The following characters are not allowed in config item 'model_name': "
                                     '/\\:*?"<>|"')

            if config['parameters']['architecture'] not in ['CNN', 'FNN']:
                raise ValueError("Config item 'residues' must be 'CNN' or 'FNN'.")

            if config['parameters']['architecture'] == 'FNN':
                if config['parameters']['multilabel'] not in ['True', 'False']:
                    raise ValueError("Config item 'multilabel' must be 'True' or 'False'.")
                if not config['parameters']['batch_size'].isdecimal():
                    raise ValueError("Config item 'batch_size' must be an integer > 0.")
            elif config['parameters']['n_layers'] not in ['2', '5', '8']:
                raise ValueError("Config item 'n_layers' must be 2, 5 or 8.")
            elif int(config['parameters']['kernel_size']) % 2 != 1:
                raise ValueError("Config item 'kernel_size' must be an odd number.")

            if not set(config['parameters']['dropout']) <= set(string.digits + '.') or \
                    not 0 <= float(config['parameters']['learning_rate']) <= 1:
                raise ValueError("Config item 'dropout' must be a float between 0 and 1.")

    if step in [2, 3, 4, 5]:
        if not config['parameters']['n_splits'].isnumeric() or not 0 < int(config['parameters']['n_splits']) < 100:
            raise ValueError("Config item 'n_splits' must be an integer between 1 and 99.")

        if step != 5:
            if step == 4:
                if config['parameters']['architecture'] == 'CNN' and \
                        not config['parameters']['balancing'] in ['', 'binary', 'binary_D', 'binary_U', 'binary_D_U']:
                    ValueError("Config item 'balancing' must be None, 'binary', 'binary_D', 'binary_U' or 'binary_D_U'"
                               ", when architecture is 'CNN'.")
                elif config['parameters']['multilabel'] == 'False' and not config['parameters']['balancing'] in \
                                                          ['', 'binary', 'binary_residues', 'binary_residues_D']:
                    ValueError("Config item 'balancing' must be None, 'binary', 'binary_D', 'binary_U', 'binary_D_U'"
                               ", 'binary_residues', 'binary_residues_D' or 'binary_residues_D_U', "
                               "when architecture is 'FNN' and 'multilabel' is False.")
            elif not config['parameters']['balancing'] in ['', 'binary', 'binary_residues',
                                                              'binary_residues_D', 'multiclass_residues']:
                ValueError("Config item 'balancing' must be None, 'binary', 'binary_residues', 'binary_residues_"
                           "D' or 'multiclass_residues'.")
    elif step in [1, 7]:
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
        else:
            raise ValueError("Config item 'annotations': too many items. Must be either FASTA-file with MobiDB "
                             "annotation, or disorder prediction (CSV) and ID (TXT) files generated by SETH")

    # for single steps
    if step == 1:
        if not exists(config['input_files']['test_set_fasta']):
            raise ValueError(f"Config item 'test_set_fasta': {config['input_files']['test_set_fasta']} is no existing "
                             f"file.")
        if not exists(config['input_files']['train_set_fasta']):
            raise ValueError(f"Config item 'train_set_fasta': {config['input_files']['train_set_fasta']} is no existing"
                             f" file.")
    elif step == 3:  # edge case, residues is tested twice, bc of restrictions in combination with architecture
        if not config['parameters']['residues'] in ['all', 'disorder_only']:
            raise ValueError("Config item 'residues' must be 'all' or 'disorder_only'.")
    elif step == 4:
        if not set(config['parameters']['learning_rate']) <= set(string.digits + '.') or \
                not 0 < float(config['parameters']['learning_rate']) < 1:
            raise ValueError("Config item 'learning_rate' must be a float between 0 and 1.")
        if not config['parameters']['patience'].isnumeric() or not 0 < int(config['parameters']['patience']) < 100:
            raise ValueError("Config item 'patience' must be an integer between 1 and 99.")
        if not config['parameters']['max_epochs'].isnumeric() or not 0 < int(config['parameters']['patience']):
            raise ValueError("Config item 'patience' must be an integer > 0.")
    elif step == 5:
        if not config['parameters']['cutoff_percent_min'].isnumeric() or \
                not 0 <= int(config['parameters']['cutoff_percent_min']) < 100:
            raise ValueError("Config item 'cutoff_percent_min' must be an integer between 0 and 99.")
        if not config['parameters']['cutoff_percent_max'].isnumeric() or \
                not 0 < int(config['parameters']['cutoff_percent_max']) <= 100 or \
                not int(config['parameters']['cutoff_percent_min']) < int(config['parameters']['cutoff_percent_max']):
            raise ValueError("Config item 'cutoff_percent_max' must be an integer between 1 and 100, and must be higher"
                             " than 'cutoff_percent_min'.")
        if not config['parameters']['step_percent'].isnumeric() or \
                not 0 < int(config['parameters']['step_percent']) <= 100:
            raise ValueError("Config item 'step_percent' must be an integer between 1 and 100.")
    elif step == 7:
        if not config['parameters']['fold'].isnumeric() or \
                not 0 <= int(config['parameters']['fold']) < int(config['parameters']['n_splits']):
            raise ValueError("Config item 'fold' must be an integer between 0 and 'n_splits'-1.")
        multilabel = config['parameters']['multilabel'] != 'False'
        if multilabel:
            pattern = re.compile('0\.[0-9]+\s*,\s*0\.[0-9]+\s*,\s*0\.[0-9]+\Z')
            if pattern.match(config['parameters']['cutoff']) is None:
                raise ValueError(
                    "Config item 'cutoff' must be a comma-separated list of 3 floats, between"
                    " 0.0 and 1.0, when 'multilabel' is True")
        else:
            pattern = re.compile('0\.[0-9]+\Z')
            if pattern.match(config['parameters']['cutoff']) is None:
                raise ValueError("Config item 'cutoff' must be a single float, between 0.0 and 1.0, when 'multilabel' "
                                 "is False")
        if config['parameters']['post_processing'] not in ['True', 'False']:
            raise ValueError("Config item 'post_processing' must be 'True' or 'False'.")
        if config['parameters']['test'] not in ['True', 'False']:
            raise ValueError("Config item 'test' must be 'True' or 'False'.")
        if config['parameters']['test'] != 'False':
            if not exists(config['input_files']['test_set_embeddings']) and not config['input_files']['test_set_embeddings'] == '':
                raise ValueError(f"Config item 'test_set_embeddings': {config['input_files']['test_set_embeddings']} "
                                 f"is no existing file.")
        if consensus_models != ['']:
            if len(consensus_folds) != len(consensus_models):
                raise ValueError(f"Config items 'consensus_models' and 'consensus_folds' must have the same number of "
                                 f"elements.")
            if not set(config['parameters']['consensus_folds']) <= set(string.digits + ',' + ' '):
                raise ValueError("Config item 'consensus_folds' must be a list of comma separated, positive integers")
            for i, m in enumerate(consensus_models):
                file = f'../results/models/binding_regions_model_{m}_fold_{str(folds[i])}.pth'
                if not exists(file):
                    raise ValueError(f"Config item 'consensus_models': {file} is no existing file.")
        if config['parameters']['test'] == 'False' and len(annotations) > 1:
            raise ValueError(f"Config item 'annotations': several input files are only allowed with 'test'==True.")


if __name__ == '__main__':
    start = datetime.now()
    dt = str(start)[:19].replace(':', '-')
    config = read_config()

    # determine workflow
    steps = config['workflow']['steps']
    if not set(steps) <= set(string.digits[1:8] + ',' + ' '):
        raise ValueError("Config item 'steps' must be a comma-separated list of digits 1...7")
    print('steps to do: ' + steps)

    # parse list parameters
    consensus_folds = config['parameters']['consensus_folds'].replace(' ', '').split(',')
    consensus_models = config['parameters']['consensus_models'].replace(' ', '').split(',')
    model_alias = {'CNN_all': 'mobidb_2_CNN_1',
                   'CNN_disorder': 'mobidb_2_D_CNN_2',
                   'FNN_all': 'mobidb_2_FNN_5',
                   'FNN_disorder': 'mobidb_2_D_FNN_4'}
    consensus_models = [model_alias[m] if m in model_alias else m for m in consensus_models]
    annotations = config['input_files']['annotations'].replace(' ', '').split(',')

    # check for parameter correctness
    for c in steps:
        if c.isdigit():
            check_config_items(int(c), config, consensus_models, consensus_folds, annotations)

    # parse bools
    wf_overwrite = config['workflow']['overwrite'] != 'False'
    param_multilabel = config['parameters']['multilabel'] != 'False'
    param_postprocessing = config['parameters']['post_processing'] != 'False'
    param_test = config['parameters']['test'] != 'False'

    # parse cutoff parameter
    cutoff = config['parameters']['cutoff'].split(',')

    # parse potential empty string to None
    oversampling = config['parameters']['balancing']
    if oversampling == '':
        oversampling = None


    if '1' in steps:
        print('step 1: preprocess dataset')
        if not wf_overwrite and \
                exists(config['input_files']['dataset_directory'] + 'test_set_input.txt') and \
                exists(config['input_files']['dataset_directory'] + 'train_set_input.txt') and \
                exists(config['input_files']['dataset_directory'] + 'test_set_stats.txt') and \
                exists(config['input_files']['dataset_directory'] + 'train_set_stats.txt'):
            print('Step 1 is skipped, all output files are already present')
        else:
            src.preprocess_dataset.preprocess(test_set_fasta=config['input_files']['test_set_fasta'],
                                              train_set_fasta=config['input_files']['train_set_fasta'],
                                              annotations=annotations,
                                              database=config['parameters']['database'],
                                              dataset_dir=config['input_files']['dataset_directory'],
                                              overwrite=wf_overwrite)
    if '2' in steps:
        print('step 2: CV and balancing')
        if not wf_overwrite:  # overwrite = False
            all_CV_splits_present = True  # check if at least 1 file is missing
            for i in range(int(config['parameters']['n_splits'])):
                if not exists(f'{config["input_files"]["dataset_directory"]}folds/CV_fold_{i}_labels_{config["parameters"]["balancing"]}.txt'):
                    all_CV_splits_present = False
                    break
        if not wf_overwrite and all_CV_splits_present:
            print('Step 2 is skipped, all output files are already present')
        else:
            src.CV_and_oversampling.split(n_splits=int(config['parameters']['n_splits']),
                                          oversampling=oversampling,
                                          dataset_dir=config['input_files']['dataset_directory'],
                                          database=config['parameters']['database']
                                          )
    if '3' in steps:
        print('step 3: sampling data points')
        if not wf_overwrite:
            all_datapoints_present = True
            for i in range(int(config['parameters']['n_splits'])):
                if 'AAindex' in config['parameters']['model_name']:
                    if not exists(f'{config["input_files"]["dataset_directory"]}folds/AAindex_representation_fold_{i}.npy'):
                        all_datapoints_present = False
                        break
                elif not exists(f'{config["input_files"]["dataset_directory"]}folds/new_datapoints_{config["parameters"]["balancing"]}_fold_{i}.npy'):
                    all_datapoints_present = False
                    break
        if not wf_overwrite and all_datapoints_present:
            print('Step 3 is skipped, all output files are already present')
        else:
            src.sampling_datapoints.sample_datapoints(train_embeddings=config['input_files']['train_set_embeddings'],
                                                      dataset_dir=config['input_files']['dataset_directory'],
                                                      database=config['parameters']['database'],
                                                      n_splits=int(config['parameters']['n_splits']),
                                                      oversampling=oversampling,
                                                      mode=config['parameters']['residues'])
    if '4' in steps:
        print('step 4: training')
        if not wf_overwrite:
            all_model_files_present = True
            for i in range(int(config['parameters']['n_splits'])):
                if not exists(f"../results/logs/training_progress_{config['parameters']['model_name']}_fold_{i}.txt"):
                    all_model_files_present = False
                    break
        if not wf_overwrite and all_model_files_present:
            print('Step 4 is skipped, all output files are already present')
        else:
            if config['parameters']['architecture'] == 'CNN':
                src.trainer_0_CNN.CNN_trainer(train_embeddings=config['input_files']['train_set_embeddings'],
                                              dataset_dir=config['input_files']['dataset_directory'],
                                              model_name=config['parameters']['model_name'],
                                              n_splits=int(config['parameters']['n_splits']),
                                              oversampling=oversampling,
                                              n_layers=int(config['parameters']['n_layers']),
                                              kernel_size=int(config['parameters']['kernel_size']),
                                              dropout=float(config['parameters']['dropout']),
                                              learning_rate=float(config['parameters']['learning_rate']),
                                              patience=int(config['parameters']['patience']),
                                              max_epochs=int(config['parameters']['max_epochs']),
                                              mode=config['parameters']['residues'])
            elif not param_multilabel:
                src.trainer_1_FNN.FNN_trainer(train_embeddings=config['input_files']['train_set_embeddings'],
                                              dataset_dir=config['input_files']['dataset_directory'],
                                              model_name=config['parameters']['model_name'],
                                              n_splits=int(config['parameters']['n_splits']),
                                              oversampling=oversampling,
                                              dropout=float(config['parameters']['dropout']),
                                              learning_rate=float(config['parameters']['learning_rate']),
                                              patience=int(config['parameters']['patience']),
                                              max_epochs=int(config['parameters']['max_epochs']),
                                              batch_size=int(config['parameters']['batch_size']),
                                              mode=config['parameters']['residues'])
            else:
                src.trainer_2_multilabel_FNN.multilabel_FNN_trainer(train_embeddings=
                                                                    config['input_files']['train_set_embeddings'],
                                                                    dataset_dir=
                                                                    config['input_files']['dataset_directory'],
                                                                    model_name=config['parameters']['model_name'],
                                                                    n_splits=int(config['parameters']['n_splits']),
                                                                    oversampling=oversampling,
                                                                    dropout=float(config['parameters']['dropout']),
                                                                    learning_rate=float(
                                                                        config['parameters']['learning_rate']),
                                                                    patience=int(config['parameters']['patience']),
                                                                    max_epochs=int(config['parameters']['max_epochs']),
                                                                    batch_size=int(config['parameters']['batch_size']))
    if '5' in steps:
        print('step 5: investigate cutoffs')
        if not wf_overwrite and exists(f"../results/logs/validation_{config['parameters']['model_name']}.txt"):
            print('Step 5 is skipped, the output file is already present')
        else:
            src.investigate_model.investigate_cutoffs(train_embeddings=config['input_files']['train_set_embeddings'],
                                                      dataset_dir=config["input_files"]["dataset_directory"],
                                                      model_name=config['parameters']['model_name'],
                                                      mode=config['parameters']['residues'],
                                                      n_splits=int(config['parameters']['n_splits']),
                                                      architecture=config['parameters']['architecture'],
                                                      multilabel=param_multilabel,
                                                      n_layers=int(config['parameters']['n_layers']),
                                                      kernel_size=int(config['parameters']['kernel_size']),
                                                      dropout=float(config['parameters']['dropout']),
                                                      batch_size=int(config['parameters']['batch_size']),
                                                      cutoff_percent_min=int(
                                                          config['parameters']['cutoff_percent_min']),
                                                      cutoff_percent_max=int(
                                                          config['parameters']['cutoff_percent_max']),
                                                      step_percent=int(config['parameters']['step_percent']))
    if '6' in steps:
        print('step 6: manual interpretation for cutoff determination')
        # TODO: make it computational

    if '7' in steps:
        print('step 7: predict')
        if not wf_overwrite:
            if param_test and \
                    exists(
                        f"../results/logs/predict_val_{config['parameters']['model_name']}_{'_'.join(cutoff)}_test.txt"):
                all_val_files_present = True
            else:
                all_val_files_present = True
                for i in range(int(config['parameters']['n_splits'])):
                    if not exists(
                            f"../results/logs/predict_val_{config['parameters']['model_name']}_{i}_{'_'.join(cutoff)}.txt"):
                        all_val_files_present = False
                        break
        if not wf_overwrite and all_val_files_present:
            print('Step 7 is skipped, all output files are already present')
        else:
            cutoff = list(map(float, cutoff))
            if len(cutoff) == 1:
                cutoff = cutoff[0]
            src.investigate_model.predict(train_embeddings=config['input_files']['train_set_embeddings'],
                                          test_embeddings=config['input_files']['test_set_embeddings'],
                                          dataset_dir=config["input_files"]["dataset_directory"],
                                          annotations=annotations,
                                          model_name=config['parameters']['model_name'],
                                          consensus_models=consensus_models,
                                          consensus_folds=consensus_folds,
                                          fold=int(config['parameters']['fold']),
                                          cutoff=cutoff,
                                          mode=config['parameters']['residues'],
                                          architecture=config['parameters']['architecture'],
                                          multilabel=param_multilabel,
                                          n_layers=int(config['parameters']['n_layers']),
                                          kernel_size=int(config['parameters']['kernel_size']),
                                          dropout=float(config['parameters']['dropout']),
                                          batch_size=int(config['parameters']['batch_size']),
                                          test=param_test,
                                          post_processing=param_postprocessing
                                          )

    # copy config file for documentation of parameters
    shutil.copyfile(src='../config.ini', dst=f"../results/logs/config_{config['parameters']['model_name']}_{dt}.ini")
    print("runtime: ", datetime.now()-start)

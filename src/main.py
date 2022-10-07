import configparser
import src.preprocess_dataset
import src.CV_and_oversampling
import src.sampling_datapoints
import src.trainer_0_CNN
import src.trainer_1_FNN
import src.trainer_2_multilabel_FNN
import src.investigate_model


# Method to read config file settings
def read_config():
    config = configparser.ConfigParser()
    config.read('./config.ini')
    return config


if __name__ == '__main__':
    config = read_config()

    # determine workflow
    steps = config['workflow']['steps']
    steps = steps.split(',')

    if '1' in steps:
        # TODO: overwrite files? for all steps
        # TODO: check parameters on correctness
        print('step 1: preprocess dataset')
        src.preprocess_dataset.preprocess(test_set_fasta=config['input_files']['test_set_fasta'],
                                          train_set_fasta=config['input_files']['train_set_fasta'],
                                          disprot_annotations=config['input_files']['disprot_annotations'])
    if '2' in steps:
        print('step 2: CV and oversampling')
        src.CV_and_oversampling.split(n_splits=int(config['parameters']['n_splits']),
                                      oversampling=config['parameters']['oversampling'])
    if '3' in steps:
        print('step 3: sampling data points')
        src.sampling_datapoints.sample_datapoints(n_splits=int(config['parameters']['n_splits']),
                                                  oversampling=config['parameters']['oversampling'],
                                                  mode=config['parameters']['residues'])
    if '4' in steps:
        print('step 4: training')
        if config['parameters']['architecture'] == 'CNN':
            src.trainer_0_CNN.CNN_trainer(model_name=config['parameters']['model_name'],
                                          n_splits=int(config['parameters']['n_splits']),
                                          oversampling=config['parameters']['oversampling'],
                                          n_layers=int(config['parameters']['n_layers']),
                                          dropout=float(config['parameters']['dropout']),
                                          learning_rate=float(config['parameters']['learning_rate']),
                                          patience=int(config['parameters']['patience']),
                                          max_epochs=int(config['parameters']['max_epochs']))
        elif not bool(config['parameters']['multilabel']):
            src.trainer_1_FNN.FNN_trainer(model_name=config['parameters']['model_name'],
                                          n_splits=int(config['parameters']['n_splits']),
                                          oversampling=config['parameters']['oversampling'],
                                          dropout=float(config['parameters']['dropout']),
                                          learning_rate=float(config['parameters']['learning_rate']),
                                          patience=int(config['parameters']['patience']),
                                          max_epochs=int(config['parameters']['max_epochs']),
                                          batch_size=int(config['parameters']['batch_size']),
                                          mode=config['parameters']['residues'])
        else:
            src.trainer_2_multilabel_FNN.multilabel_FNN_trainer(model_name=config['parameters']['model_name'],
                                                                n_splits=int(config['parameters']['n_splits']),
                                                                oversampling=config['parameters']['oversampling'],
                                                                dropout=float(config['parameters']['dropout']),
                                                                learning_rate=float(
                                                                    config['parameters']['learning_rate']),
                                                                patience=int(config['parameters']['patience']),
                                                                max_epochs=int(config['parameters']['max_epochs']),
                                                                batch_size=int(config['parameters']['batch_size']))
    if '5' in steps:
        print('step 5: investigate cutoffs')
        src.investigate_model.investigate_cutoffs(model_name=config['parameters']['model_name'],
                                                  mode=config['parameters']['residues'],
                                                  n_splits=int(config['parameters']['n_splits']),
                                                  architecture=config['parameters']['architecture'],
                                                  multilabel=bool(config['parameters']['multilabel']),
                                                  n_layers=int(config['parameters']['n_layers']),
                                                  dropout=float(config['parameters']['dropout']),
                                                  batch_size=int(config['parameters']['batch_size']),
                                                  cutoff_percent_min=int(config['parameters']['cutoff_percent_min']),
                                                  cutoff_percent_max=int(config['parameters']['cutoff_percent_max']),
                                                  step_percent=int(config['parameters']['step_percent']))
    if '6' in steps:
        print('step 6: manual interpretation for cutoff determination')
        # TODO: make it computational

    if '7' in steps:
        print('step 7: predict')

        # parse cutoff parameter
        cutoff = config['parameters']['cutoff'].split(',')
        list(map(float, cutoff))
        if len(cutoff) == 1:
            cutoff = cutoff[0]

        src.investigate_model.predict(model_name=config['parameters']['model_name'],
                                      fold=int(config['parameters']['fold']),
                                      cutoff=cutoff,
                                      mode=config['parameters']['residues'],
                                      architecture=config['parameters']['architecture'],
                                      multilabel=bool(config['parameters']['multilabel']),
                                      n_layers=int(config['parameters']['n_layers']),
                                      dropout=float(config['parameters']['dropout']),
                                      batch_size=int(config['parameters']['batch_size']),
                                      test=bool(config['parameters']['test']),
                                      post_processing=bool(config['parameters']['post_processing'])
                                      )

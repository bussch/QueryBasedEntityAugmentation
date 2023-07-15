import random

from local.sender_idf_baseline import SenderIDFBaseline
from local.sender_dataset_level import SenderDatasetLevel
from local.sender_hybrid import SenderHybrid
from local.sender_longformer import SenderLongformer
from local.sender_llama import SenderLlama
from external.receiver import Receiver
from data_manage.oracle import Oracle
import multiprocessing as mp
import shutil
import pickle as pickle
import os, sys
from additionalScripts.output_config import OutputConfig
import numpy as np


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def experiment(experiment_config, sender, oracle, receiver, average_run=1, seed=None):
    def get_matches(id_local, result_list):
        return [(id_external, 1 / (rank + 1), receiver.signalIndex[id_external]) for rank, id_external in
                enumerate(result_list)
                if oracle.isMatch(id_local, id_external)]

    if sender.config['distribution'] == 'zipfian':
        if seed is None:
            print('Generating seed for zipfian')
            seed = random.randrange(sys.maxsize)
        else:
            print('Using ' + str(seed) + ' seed for zipfian')
        sender.generateZipfianDistribution(seed)
    elif sender.config['distribution'] == 'fixed':
        # with open(f"fixedDistributions/fixedDist_{experiment_config['dataset_name']}-{experiment_config['interactions']}-{average_run-1}", 'rb') as file:
        with open(f"fixedDistributions/fixedDist_{experiment_config['dataset_name']}-10000-{average_run - 1}",
                  'rb') as file:
            sender.tuple_series = pickle.load(file)
            print(f'Loaded: {sender.tuple_series[:10]}')
    elif sender.config['distribution'] == 'uniform_fixed':
        with open(
                f"fixedDistributions/fixedDist_uniform_{experiment_config['dataset_name']}-{experiment_config['interactions']}-{average_run - 1}",
                'rb') as file:
            sender.tuple_series = pickle.load(file)
            print(f'Loaded (uniform fixed): {sender.tuple_series[:10]}')

    queryLog = list()

    # Statistics
    mostRecentRR = list()
    mostRecentPrecision = list()
    mostRecentRecall = list()

    # Model data
    mostRecentLoss = list()

    for interaction_num in range(1, experiment_config['interactions'] + 1):
        '''
                    Pick tuple in local, generate query based on tuple and retrieve results from external
        '''
        id_local = sender.pickTupleToJoin()
        querySignals = sender.generate_query(id_local, experiment_config['query_length'])
        tuplesReturnedInOrder = receiver.returnTuples(querySignals, experiment_config['top_k_size'])

        '''
                    Check external results for matches
        '''
        matches = get_matches(id_local, tuplesReturnedInOrder)
        reciprocal_rank = 0
        if len(matches) > 0:
            reciprocal_rank = matches[0][1]  # RR is the RR of the highest ranked match

        '''
                    Update Local
        '''
        # Produces [[rr_11, ..., rr_1n], ... [rr_m1, ..., rr_mn]] where m is the amount of matches and n is the amount
        # of terms in the query. EX: if the first term of the query is in the first match (which has a rr of .5), then
        # rr_11 will be .5.
        y = []
        for match in matches:
            terms_in_match = [term.keyword for term in match[2]]
            y.append([match[1] if signal[0].keyword in terms_in_match else 0 for signal in querySignals])
        x = [pair[0] for pair in querySignals]
        if len(matches) > 0:
            y = np.average(y, axis=0)
        else:
            y = np.zeros(len(querySignals))

        if len(matches) != 0:
            sender.update_tuple_specific_features(id_local, [match[0] for match in matches],
                                                  receiver.signalIndex)

        # Update model
        loss = sender.update_model(id_local, x, y)

        '''
                    Update tuple statistics
        '''
        precision = len(matches) / max(len(tuplesReturnedInOrder), 1)
        recall = len(matches) / max(oracle.getTotalTrue(id_local), 1)

        mostRecentRR.append(reciprocal_rank)
        mostRecentRecall.append(recall)
        mostRecentPrecision.append(precision)
        mostRecentLoss.append(loss)

        queryLog.append((id_local, querySignals, reciprocal_rank))

        '''
                    Print/Save statistics
        '''
        print_every = 500
        if interaction_num % print_every == 0:
            print(experiment_config['dataset_name'] + '_' + str(interaction_num) + '_returned='
                  + str(experiment_config['top_k_size']) + '_sent=' + str(experiment_config['query_length'])
                  + '_AverageRun=' + str(average_run))
            print('MRR: ' + str(sum(mostRecentRR) / interaction_num))
            print(f'Avg query length: {sum([len(log_entry[1]) for log_entry in queryLog]) / interaction_num}')
            print()

        if (interaction_num % experiment_config['interactions'] == 0):
            zip_dir = experiment_config['save_path'] + '/' + str(interaction_num) + '_seed=' + str(
                seed) + '_AverageRun=' + str(average_run)
            os.makedirs(zip_dir)

            filePre = zip_dir

            print('saving')
            # del sender.signalIndex
            # del sender.idf
            if hasattr(sender, 'dataset'):
                del sender.dataset.table
            if str(sender) == 'longformer':
                del sender.model
            if hasattr(sender, '_external_signalIndex'):
                del sender._external_signalIndex

            # save_obj(sender, '{}/sender'.format(filePre))
            # save_obj(receiver, '{}/receiver_{}'.format(filePre))

            save_obj(mostRecentRR, '{}/mrr'.format(filePre))
            save_obj(mostRecentRecall, '{}/recall'.format(filePre))
            save_obj(mostRecentPrecision, '{}/precision'.format(filePre))
            save_obj(mostRecentLoss, '{}/loss'.format(filePre))
            save_obj(queryLog, '{}/query_log'.format(filePre))

            shutil.make_archive(zip_dir, 'zip', zip_dir)
            shutil.rmtree(zip_dir)


def get_dir_string(config_dict, keys_in_order):
    def _shorten(str):
        split_str = str.split('_')
        new_str = ''
        for i in range(len(split_str)):
            new_str += split_str[i][:2]
            if i != len(split_str) - 1:
                new_str += '_'
        return new_str

    return '-'.join(['{}={}'.format(_shorten(key), config_dict[key]) for key in keys_in_order if key in config_dict])


if __name__ == '__main__':

    # Experiment parameters
    dataset_name = 'google'
    dataset_path = 'datasets/'
    experiment_config = {
        'average_runs': 1,
        # Amount of experiments to perform. Note that all processes but one will crash if data structures have not
        # been created already (the first process will create them and the rest will crash trying to read them).
        # Just run with 1 process when first creating data.
        'interactions': 2000,  # interactions to run (i.e., amount of times we query the external)
        'top_k_size': 20,  # top-k tuples returned from external
        'query_length': 4,  # keywords to send (length of query)
        'dataset_name': dataset_name
    }

    # Receiver setup
    receiver_config = {
        'config_path': 'output_path_config',
        'data_file_path': dataset_path + dataset_name + '/target.csv',
        'dataset_name': dataset_name
    }
    receiver = Receiver(receiver_config)

    # Sender parameters
    sender_config = {
        # Shared parameters
        'distribution': 'zipfian',  # distribution used to select intents to find matches for
        'config_path': 'output_path_config',
        'data_file_path': dataset_path + dataset_name + '/source.csv',
        'dataset_name': dataset_name,

        # Dataset-Level
        'unsupervised_term_borrowing': False,
        'external_feat_specific_unsupervised': False,
        'p_thresh': None,  # Dynamic query length. If None, then not used.
        # Otherwise, specify probability mass threshold (0, 1).

        # Dataset-Level/Hybrid parameters
        'alpha': 0.2,
        'external_feat_specific': False,
        'supervised_term_borrowing': False,

        # Hybrid parameters
        'window_size': 50,
        'rr_threshold': 1 / 15,  # Rank 15 or worse

        # Longformer/Llama parameters
        'epsilon': 0.05,  # e-greedy exploration
        'buffer_size': 50,  # amount of previous (x,y) update samples to keep in LIFO queue
        'buffer_sample_size': 8,  # amount of (x,y) pairs to uniformly sample from LIFO queue when updating
        'split_sample': True  # Accumulates gradients one sample at a time. Can use if GPU lacks enough memory.
    }
    # sender = SenderIDFBaseline(sender_config, receiver)
    sender = SenderDatasetLevel(sender_config, receiver)
    # sender = SenderHybrid(sender_config, receiver)
    # sender = SenderLongformer(sender_config, receiver)
    # sender = SenderLlama(sender_config, receiver)

    # Oracle setup
    oracle = Oracle(dataset_path + dataset_name + '/ground_truth.csv')

    # Filepath for where experiment results will be saved "<config_path>/<experiment_details>"
    # Add additional configuration settings here to distinguish between experiments run
    experiment_config['save_path'] = OutputConfig('output_path_config').paths['experiment_results'] + \
                                     get_dir_string(experiment_config, ['dataset_name', 'query_length', 'interactions',
                                                                        'distribution']) + '/' + \
                                     str(sender) + '-' + get_dir_string(sender_config,
                                                                        ['alpha', 'external_feat_specific',
                                                                         'unsupervised_term_borrowing',
                                                                         'supervised_term_borrowing', 'p_thresh'])

    # remove entities that have no match in any external source
    if dataset_name == 'chebi' or dataset_name == 'drugcentral':
        findable_entities = set(oracle.getAllSourceMatches())
        for unfindable_entity in set(sender.signalIndex.keys()).difference(findable_entities):
            del sender.signalIndex[unfindable_entity]

    if not os.path.exists(experiment_config['save_path']):
        print(experiment_config['save_path'] + " does not exist.")
        print('Averaged runs: {}'.format(experiment_config['average_runs']))
        print('creating: ' + experiment_config['save_path'])
        os.makedirs(experiment_config['save_path'])
    else:
        print(experiment_config['save_path'] + " exists.")
        print('Averaged runs: {}'.format(experiment_config['average_runs']))

    process_list = []
    for i in range(experiment_config['average_runs']):
        p = mp.Process(target=experiment, args=(experiment_config, sender, oracle, receiver, i + 1))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    print('Complete.')

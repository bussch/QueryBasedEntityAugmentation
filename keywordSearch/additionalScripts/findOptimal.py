import os
import pickle
import multiprocessing as mp

from additionalScripts.utils.calculateOptimalQueries import calculateOptimalQueries
from external.receiver import Receiver
from local.sender_base import SenderBase
from data_manage.oracle import Oracle


if __name__ == '__main__':
    PATH_TO_DATASET = ''
    dataset_path = 'datasets/'
    dataset_name = 'chebi'

    # Receiver setup
    receiver_list = {}
    receiver_config = {'config_path': 'output_path_config'}
    receiver_config['data_file_path'] = dataset_path + dataset_name + '/target.csv'
    receiver_config['dataset_name'] = dataset_name
    receiver = Receiver(receiver_config)

    # Sender parameters. Most of these settings are not applicable to this brute force script,
    # but are otherwise required to instantiate the model (so they appear here anyways)
    sender_config = {
        # Shared parameters
        'distribution': 'uniform',
        'dataset_name': dataset_name,
        'data_file_path': dataset_path + dataset_name + '/source.csv',
        'external_feat_specific': False,
        'supervised_term_borrowing': False,
        'config_path': 'output_path_config',
    }

    if dataset_name == 'drugs':
        sender_config['data_file_path'] = dataset_path + dataset_name + '/' + local_datasource + '.csv'
    sender = SenderBase(sender_config, receiver_list)

    # Oracle setup
    oracle = Oracle(dataset_path + dataset_name + '/ground_truth.csv')

    # load sample list (a pickled python list of local entity IDs for datset)
    sample_list = pickle.load(open(f'samplelist.pkl', 'rb'))

    save_path = f'/data/experiments/bestQ/{dataset_name}'
    if not os.path.exists(save_path):
        print(f"{save_path} does not exist")
        print(f"creating {save_path}")
        os.makedirs(save_path)
    else:
        print(f"{save_path} exists, using it")

    K = 20
    MIN = 1
    MAX = 3
    PROCESSES = 1
    SEGMENT_OFFSET = 0
    SEGMENT_LENGTH = len(sample_list)
    BREAK_PART = [1, 1]  # [x, y] x of y parts
    PART_LENGTH = SEGMENT_LENGTH / BREAK_PART[1]
    PART_START = 0
    PART_END = 1000

    print('Here: {}'.format(SEGMENT_LENGTH))
    print('{} - {}'.format(PART_START, PART_END))

    thread_list = []

    print('Starting {} threads... on {} with queries of size {}-{}'.format(PROCESSES, dataset_name, MIN, MAX))

    keys_per_thread = int((PART_END - PART_START) / PROCESSES)
    for i in range(PROCESSES):
        offset = PART_START + (i * keys_per_thread)
        if i == PROCESSES - 1:
            t = mp.Process(target=calculateOptimalQueries,
                           args=(dataset_name, sender, receiver, oracle, offset, PART_END, MIN, MAX, save_path, K, sample_list))
        else:
            print()
            t = mp.Process(target=calculateOptimalQueries,
                           args=(dataset_name, sender, receiver, oracle, offset, offset + keys_per_thread, MIN, MAX, save_path, K, sample_list))
        t.start()
        thread_list.append(t)

    for t in thread_list:
        t.join()
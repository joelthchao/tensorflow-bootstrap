import argparse
import json
import os

import tensorflow as tf

from bootstrap.datasets.mnist_dataset import MnistDataset
from bootstrap.models.mlp import MLP
from bootstrap.trainers.basic_trainer import BasicTrainer
from bootstrap.utils import make_path, pprint_params


def main(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # configure path
    make_path(args.project_path)
    checkpoint_path = make_path(os.path.join(args.project_path, 'checkpoints'))

    # params
    params = prepare_params(args)

    # graph and session
    graph = tf.Graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(graph=graph, config=tf_config)

    # dataset
    train_dataset = MnistDataset(params, mode='train')
    train_dataset.build()
    test_dataset = MnistDataset(params, mode='test')
    test_dataset.build()

    # model
    model = MLP(session, graph, params)
    model.build()

    # training
    trainer = BasicTrainer(session, graph, model, train_dataset, test_dataset, params)
    trainer.train()


def prepare_params(args):
    # params and exp id
    params = {
        'batch_size': 32,
        'num_class': 10,
        'hidden_size': [64, 64],
        'activation': 'relu',
        'input_size': 28 * 28,
        'total_step': 1000,
        'display_step': 200,
        'val_step': 1000,
    }
    param_file = os.path.join(args.project_path, 'params.json')
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            extra_params = json.load(f)
        params.update(extra_params)

    with open(param_file, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)
    pprint_params(params)

    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train mnist for demo')
    parser.add_argument('--project_path', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='')
    args = parser.parse_args()

    main(args)

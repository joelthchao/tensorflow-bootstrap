from collections import defaultdict
from itertools import islice
import time

import numpy as np
import tensorflow as tf

from bootstrap.models.base_model import BaseModel
from bootstrap.datasets.base_dataset import BaseDataset
from bootstrap.utils import summarize_metrics


class BasicTrainer:
    def __init__(self, session: tf.Session, graph: tf.Graph, model: BaseModel,
                 train_dataset: BaseDataset, val_dataset: BaseDataset, params: dict):
        self.session = session
        self.graph = graph
        self.model = model
        self.params = params
        self.train_dataset = train_dataset
        self.train_batch_gen = train_dataset.batch_generator()
        self.val_dataset = val_dataset
        self.params = params

    def train(self):
        start_steps = self.session.run(self.model.global_step)
        losses = []
        metrics_dict = defaultdict(list)
        step_times = []
        for step in range(start_steps, self.params['total_step']):
            start_time = time.time()
            loss, metrics = self.train_step()
            losses.append(loss)
            for k, v in metrics:
                metrics_dict[k].append(v)
            step_times.append(time.time() - start_time)

            if step % self.params['display_step'] == 0:
                mean_loss = np.mean(losses)
                mean_step_time = np.mean(step_times)
                metric_str = summarize_metrics(metrics_dict)
                print('[Train] Step {:6d} - Loss: {:.4f} - {} - {:.4f}s/step'.format(
                    step, mean_loss, metric_str, mean_step_time))
                losses = []
                metrics_dict = defaultdict(list)
                step_times = []

            if step % self.params['val_step'] == 0:
                self.val_step(step)

    def train_step(self):
        batch_data = next(self.train_batch_gen)
        feed_dict = self.model.make_feed_dict(batch_data)
        if self.model.metrics:
            keys = sorted(self.model.metrics.keys())
            ops = [self.model.train_op, self.model.loss] + [self.model.metrics[k] for k in keys]
            _, loss, metric_values = self.session.run(ops, feed_dict=feed_dict)
            metrics = [(k, v) for k, v in zip(keys, metric_values)]
        else:
            metrics = []
            _, loss = self.session.run(
                [self.model.train_op, self.model.loss], feed_dict=feed_dict)

        return loss, metrics

    def val_step(self, step, num_batches=200):
        losses = []
        metrics_dict = defaultdict(list)
        for batch_data in islice(self.val_dataset.batch_generator(), num_batches):
            feed_dict = self.model.make_feed_dict(batch_data)
            if self.model.metrics:
                keys = sorted(self.model.metrics.keys())
                ops = [self.model.loss] + [self.model.metrics[k] for k in keys]
                loss, metric_values = self.session.run(ops, feed_dict=feed_dict)
                metrics = [(k, v) for k, v in zip(keys, metric_values)]
            else:
                metrics = []

            loss = self.session.run(self.model.loss, feed_dict=feed_dict)
            losses.append(loss)
            for k, v in metrics:
                metrics_dict[k].append(v)

        mean_loss = np.mean(losses)
        metric_str = summarize_metrics(metrics_dict)
        print('[  Val] Step {:6d} - Loss: {:.4f} - {}'.format(step, mean_loss, metric_str))

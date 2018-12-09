import tensorflow as tf


class BaseModel:
    def __init__(self, session: tf.Session, graph: tf.Graph, params: dict):
        self.session = session
        self.graph = graph
        self.params = params

        with self.graph.as_default():
            self.global_step = tf.Variable(1, trainable=False, name='global_step')

        self.train_op = None
        self.loss = None
        self.optimizer = None
        self.metrics = {}  # name: tensor

    def save(self):
        # TODO: Implement
        pass

    def load(self, path, name):
        # TODO: Implement
        pass

    def init_saver(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=self.params['max_checkpoint_to_keep'])

    def build(self):
        raise NotImplementedError

    def make_feed_dict(self, batch_data: dict):
        raise NotImplementedError

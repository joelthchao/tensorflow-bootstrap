import tensorflow as tf
from bootstrap.models.base_model import BaseModel


ACTIVATION_MAP = {
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
}


class MLP(BaseModel):
    def __init__(self, session, graph, params):
        super().__init__(session, graph, params)
        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.num_class = params['num_class']
        self.activation = params['activation']

    def build(self):
        with self.graph.as_default():
            with tf.name_scope('input'):
                self.x = tf.placeholder(tf.float32, (None, self.input_size), name='x')
                self.y = tf.placeholder(tf.int32, (None, self.num_class), name='y')

            with tf.name_scope('hidden'):
                net = self.x
                activaction = ACTIVATION_MAP.get(self.activation)
                for h in self.hidden_size:
                    net = tf.layers.dense(net, h, activation=activaction)

            with tf.name_scope('output'):
                logits = tf.layers.dense(net, self.num_class)
                acc = tf.equal(tf.arg_max(logits, 1), tf.arg_max(self.y, 1))
                self.metrics['acc'] = acc

            with tf.name_scope('optimize'):
                self.loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=self.y)
                self.optimizer = tf.train.AdamOptimizer(name='optimizer')
                train_vars = list(tf.trainable_variables())
                grad_vars = self.optimizer.compute_gradients(self.loss, var_list=train_vars)
                self.train_op = self.optimizer.apply_gradients(
                    grad_vars, global_step=self.global_step, name='train_op')

            self.init = tf.global_variables_initializer()
            self.session.run(self.init)

    def make_feed_dict(self, batch_data):
        return {
            self.x: batch_data['x'],
            self.y: batch_data['y'],
        }

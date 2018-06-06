import numpy as np
from time import time
from collections import OrderedDict
from typing import List, Callable
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras


def _make_layer_func(base_layer_func, base_name: str):
    """Return a function which can be used to create layers with unique names."""
    i = -1
    def layer_func(*args, **kwargs):
        nonlocal i
        i += 1
        return base_layer_func(*args, name=f"{base_name}_{i}", **kwargs)
    return layer_func


N_FILTERS = 128
PADDING = 'same'
ACTIVATION = 'relu'

# technically its 1D, so not 1x1, but I find 1x1 reminds me better what it's doing
_CONV_1x1 = _make_layer_func(keras.layers.Conv1D, 'Conv_1x1')
CONV_1x1 = lambda: _CONV_1x1(N_FILTERS, kernel_size=1, padding=PADDING, activation=ACTIVATION)

# For the reduce block, we'll just set the stride of the operations used on the inputs to 2,
# so each "op" here is actually a function that returns the desired layer
_CONV_OPS = [
    _make_layer_func(keras.layers.Conv1D, 'Conv3'),
    _make_layer_func(keras.layers.Conv1D, 'Conv5'),
    _make_layer_func(keras.layers.Conv1D, 'Conv7'),
    _make_layer_func(keras.layers.Lambda, 'Id')
]

CONV_OPS = [
    lambda s: _CONV_OPS[0](N_FILTERS, kernel_size=3, strides=s, padding=PADDING, activation=ACTIVATION),
    lambda s: _CONV_OPS[1](N_FILTERS, kernel_size=5, strides=s, padding=PADDING, activation=ACTIVATION),
    lambda s: _CONV_OPS[2](N_FILTERS, kernel_size=7, strides=s, padding=PADDING, activation=ACTIVATION),
    lambda s: _CONV_OPS[3](lambda x: x),
    lambda s: keras.layers.MaxPool1D(strides=s, padding='same'),
    lambda s: keras.layers.AvgPool1D(strides=s, padding='same')
]

COMBINATION_METHODS = {
    'add': _make_layer_func(lambda name: keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=0), name=name), 'Add'),
    'mult': _make_layer_func(lambda name: keras.layers.Lambda(lambda x: tf.reduce_prod(x, axis=0), name=name), 'Mult')
}



class Node:
    def __init__(self, inputs, ops, combination_method):
       self.inputs = inputs
       self.ops = ops
       self.combination_method = combination_method


class Block(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class ArchitectureSampler:
    def __init__(self,
                 n_ops_per_node,
                 n_nodes_per_block,
                 n_block_repeats,
                 n_blocks_between_reduce,
                 combination_methods,
                 **unused_kwargs):
        self.n_ops_per_node = n_ops_per_node
        self.n_nodes_per_block = n_nodes_per_block
        self.n_block_repeats = n_block_repeats
        self.n_blocks_between_reduce = n_blocks_between_reduce
        self.combination_methods = [COMBINATION_METHODS[m] for m in combination_methods]

    def _sample_node(self, all_inputs, unused_outputs: set, stride: int=1):
        inputs = np.random.choice(all_inputs, size=self.n_ops_per_node, replace=True)
        combination_method = np.random.choice(self.combination_methods)
        ops = np.random.choice(CONV_OPS, size=self.n_ops_per_node, replace=True)

        unused_outputs.difference_update(inputs)

        outputs = []
        for i in range(len(inputs)):
            input_ = inputs[i]

            # only use strided operations on the original inputs so there's only one reduction
            if input_.shape[1] != all_inputs[0].shape[1]:
                op = ops[i](1)
            else:
                op = ops[i](stride)

                # don't allow Id op to be used on the original input; that needs to be reduced
                while 'Id' in op.name:
                    op = np.random.choice(CONV_OPS)(stride)

            if input_.shape[-1] != N_FILTERS and 'Pool' in str(type(op)):
                input_ = self._input_conv_1x1

            outputs.append(op(input_))

        return combination_method()(outputs)

    def _sample_block(self, inputs, seed: int, stride: int=1):
        np.random.seed(seed)
        unused_outputs = set()
        for i in range(self.n_nodes_per_block):
            with tf.variable_scope(f'node_{i}'):
                output = self._sample_node(inputs, unused_outputs, stride=stride)
            inputs.append(output)
            unused_outputs.add(output)

        unused_outputs = list(unused_outputs)
        if len(unused_outputs) > 1:
            return CONV_1x1()(keras.layers.concatenate(unused_outputs, axis=-1))
        else:
            return unused_outputs[0]

    def sample_block(self) -> Callable:
        seed = np.random.randint(1_000_000)
        return lambda inputs: self._sample_block(inputs, seed)

    def sample_reduce_block(self) -> Callable:
        seed = np.random.randint(1_000_000)
        return lambda inputs: self._sample_block(inputs, seed, stride=2)

    def sample_arch(self, input_shape, n_arches: int=1) -> List:
        arches = []
        for _ in range(n_arches):
            input_ = keras.layers.Input(input_shape)
            # use the input twice because later layers will have the two previous
            # layers' outputs as possible inputs (except after a reduction)
            inputs = [input_, input_]
            self._input_conv_1x1 = CONV_1x1()(input_)

            block = self.sample_block()
            reduce_block = self.sample_reduce_block()

            # you can also select the input of the last block as input to a node
            # in the next block (as long as there was no reduce in between)
            last_block_input = input_
            for i in range(self.n_block_repeats):
                with tf.variable_scope(f'repeat_{i}'):
                    for j in range(self.n_blocks_between_reduce):
                        with tf.variable_scope(f'block_{j}'):
                            block_output = block(inputs)
                        inputs = [last_block_input, block_output]
                        last_block_input = block_output

                    # don't add a reduction block just before the output layer
                    if i != self.n_block_repeats - 1:
                        with tf.variable_scope(f'reduce'):
                            block_output = reduce_block(inputs)
                        inputs = [block_output, block_output]
                        last_block_input = block_output

            with tf.variable_scope('output'):
                output = keras.layers.Flatten()(block_output)
                output = keras.layers.Dense(1)(output)

            arches.append(Model(keras.models.Model(input_, output)))
        return arches


class WeightsLoader:
    # TODO: do we need to process the names at all?
    # yes... because each block will have something like 'repeat_0/block_0/node_0/Conv5_0/kernel:0'
    # we just want to know node_0/op/input/[kernel or bias]
    def __init__(self, weights_file: str):
        """
        :param weights_file: should end in .npz
        """
        self.weights_file = weights_file
        self._load_weights_from_disk()

    def _load_weights_from_disk(self):
        self.weights = dict(np.load(self.weights_fname))

    def save_weights_to_disk(self):
        np.savez_compressed(self.weights_file, **self.weights)

    def load_weights(self, model: tf.keras.models.Model):
        """Load weights into `model` by name."""
        weights = model.get_weights()
        weights = OrderedDict((model.trainable_weights[i].name, weights[i]) for i in range(len(weights)))

        # override random weights with the pre-trained ones
        for weight_name in weights:
            if weight_name in self.weights:
                weights[weight_name] = self.weights[weight_name]

        model.set_weights(list(weights.values()))

    def save_weights(self, model: tf.keras.models.Model):
        """Update in-memory weights from `model`."""
        weights = model.get_weights()
        weights = {model.trainable_weights[i].name: weights[i] for i in range(len(weights))}

        for weight_name in weights:
            if weight_name in self.weights:
                self.weights[weight_name] = weights[weight_name]


class Model:
    def __init__(self, keras_model: keras.models.Model):
        self.model = keras_model
        self._fitness = None

    def train(self):
        self.model.compile()
        self.model.fit()

        self._fitness = 0
        self.trained_at = time()

    @property
    def fitness(self):
        if self._fitness is None:
            self.train()
        return self._fitness

    @property
    def age(self):
        return time() - self.trained_at

    def make_child(self):
        child = Model(self.arch)
        child.train()
        return child


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-no', '--n_ops_per_node', type=int, default=2)
    parser.add_argument('-nn', '--n_nodes_per_block', type=int, default=5)
    parser.add_argument('-nb', '--n_block_repeats', type=int)
    parser.add_argument('-nr', '--n_blocks_between_reduce', type=int)
    parser.add_argument('-cm', '--combination_methods', nargs='+', default=['add'],
                        help='One or more of {add, mult}; which elementwise operation to use'
                              'when combing the ops for a node (if add and mult are both given,'
                              'one of them will be selected to use for each node).')
    parser.add_argument('-nf', '--n_filters', type=int,
                        help='The same number of filters is used throughout the network.')
    parser.add_argument('-na', '--n_arches', type=int, help='Number of architectures in population.')
    parser.add_argument('-ns', '--n_sample', type=int, help='Number of models in a sample.')
    parser.add_argument('-ng', '--n_generations', type=int)
    parser.add_argument('-p', '--experiment_path', help='Location to save all experiment data.')
    parser.add_argument('-k', '--comet_key')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    regularized_evolution = True  # remove oldest arch from sample instead of worst

    input_shape = () # TODO

    # set up the initial population
    arch_sampler = ArchitectureSampler(input_shape=input_shape, **dict(args))
    models = arch_sampler.sample_arch(input_shape, n_arches=args.n_arches)

    for generation in range(args.n_generations):
        # select a random sample from the population; mutate the best, remove the oldest or worse
        sample_models = np.random.choice(models, size=args.n_sample, replace=False)
        best_model = max(sample_models, key=lambda model: model.fitness)
        new_model = best_model.make_child()
        models.append(new_model)

        if regularized_evolution:
            worst = max(sample_models, key=lambda model: model.age)
        else:
            worst = min(sample_models, key=lambda model: model.fitness)

        models.remove(worst)

        if args.verbose:
            mean_fitness = sum(m.fitness for m in models) / len(models)
            print(f'Generation {generation}: Removed model with fitness = {worst.fitness}, '
                  f'added model with fitness = {new_model.fitness} (mean fitness = {mean_fitness}).')
            # exp.log..('mean_fitness', mean_fitness)

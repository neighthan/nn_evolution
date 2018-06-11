# from future import __annotations__
import numpy as np
from time import time
from collections import OrderedDict
from typing import List, Tuple, Sequence, Callable, Any
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
from evo.utils import check_blocks

N_FILTERS = 128
PADDING = 'same'
ACTIVATION = 'relu'

# technically its 1D, so not 1x1, but I find 1x1 reminds me better what it's doing
CONV_1x1 = lambda: keras.layers.Conv1D(N_FILTERS, kernel_size=1, padding=PADDING, activation=ACTIVATION)

# For the reduce block, we'll just set the stride of the operations used on the inputs to 2,
# so each "op" here is actually a function that returns the desired layer
CONV_OPS = [
    lambda s: keras.layers.Conv1D(N_FILTERS, kernel_size=3, strides=s, padding=PADDING, activation=ACTIVATION),
    lambda s: keras.layers.Conv1D(N_FILTERS, kernel_size=5, strides=s, padding=PADDING, activation=ACTIVATION),
    lambda s: keras.layers.Conv1D(N_FILTERS, kernel_size=7, strides=s, padding=PADDING, activation=ACTIVATION),
    lambda s: keras.layers.MaxPool1D(strides=s, padding='same'),
    lambda s: keras.layers.AvgPool1D(strides=s, padding='same'),
    # Id *must* be the last op; we make sure not to use the last op in the input when reducing
    lambda s: keras.layers.Lambda(lambda x: x)
]

COMBINATION_METHODS = {
    'add': lambda: keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=0)),
    'mult': lambda: keras.layers.Lambda(lambda x: tf.reduce_prod(x, axis=0))
}


class Node:
    def __init__(self, inputs, ops, combination_method):
       self.inputs = inputs
       self.ops = ops
       self.combination_method = combination_method


class Block:
    def __init__(self,
                 n_ops_per_node: int,
                 n_nodes: int,
                 node_input_idx: List[List[int]],
                 node_op_idx: List[List[int]],
                 node_combination_methods: List[str],
                 stride: int
                 ):
        self.n_ops_per_node = n_ops_per_node
        self.n_nodes = n_nodes
        self.node_input_idx = node_input_idx
        self.node_op_idx = node_op_idx
        self.node_combination_methods = node_combination_methods
        self.stride = stride

    @classmethod
    def sample_block(cls, n_ops_per_node: int, n_nodes: int, combination_methods: List[str], stride: int=1):  # -> Block
        """Randomly sample connections and ops for a `Block`, which is returned"""
        node_op_idx = []
        node_input_idx = []
        node_combination_methods = []

        inputs = [0, 1]
        conv_op_idx = list(range(len(CONV_OPS)))

        for i in range(n_nodes):
            input_idx = np.random.choice(inputs, size=n_ops_per_node, replace=True)
            # sort so that we don't have different weight matrices for the same thing like
            # node1/inp1_op1_inp2_op2 and node1/inp2_op2_inp1_op1
            node_input_idx.append(sorted(input_idx))

            node_combination_methods.append(np.random.choice(combination_methods))

            if stride == 1:
                op_idx = np.random.choice(conv_op_idx, size=n_ops_per_node, replace=True)
            else:  # this is a reduce block
                op_idx = []
                for input_i in input_idx:
                    op = np.random.choice(conv_op_idx)

                    # the first two inputs are the ones that need to be strided on
                    # so we can't use an id op, which won't reduce
                    if input_i in [0, 1]:
                        while op == len(CONV_OPS) - 1:  # Id is the last CONV_OP
                            op = np.random.choice(conv_op_idx)

                    op_idx.append(op)

            node_op_idx.append(op_idx)

        return cls(n_ops_per_node, n_nodes, node_input_idx, node_op_idx, node_combination_methods, stride)

    def _build_node(self, inputs, unused_outputs: set, node_idx: int):
        node_ops = self.node_op_idx[node_idx]
        node_inputs = [inputs[idx] for idx in self.node_input_idx[node_idx]]
        combination_method = self.node_combination_methods[node_idx]

        unused_outputs.difference_update(node_inputs)

        outputs = []
        for i in range(len(node_inputs)):
            input_ = node_inputs[i]

            # only use strided operations on the original inputs so there's only one reduction
            if input_.shape[1].value != inputs[0].shape[1].value:
                op = CONV_OPS[node_ops[i]](1)
            else:
                op = CONV_OPS[node_ops[i]](self.stride)

            if input_.shape[-1] != N_FILTERS and ('Pool' in str(type(op)) or node_ops[i] == len(CONV_OPS) - 1):
                input_ = self._input_conv_1x1

            # we allow for a node to use the same op on the same input each time; the op
            # will be parametrized differently for each one.
            with tf.variable_scope(f'inp_{i}_{self.node_input_idx[node_idx][i]}'):
                with tf.variable_scope(f'op_{node_ops[i]}'):
                    outputs.append(op(input_))

        return COMBINATION_METHODS[combination_method]()(outputs)

    def _build_block(self, inputs):
        unused_outputs = set()
        for i in range(self.n_nodes):
            with tf.variable_scope(f'node_{i}'):
                output = self._build_node(inputs, unused_outputs, node_idx=i)
            inputs.append(output)
            unused_outputs.add(output)

        unused_outputs = list(unused_outputs)
        if len(unused_outputs) > 1:
            return CONV_1x1()(keras.layers.concatenate(unused_outputs, axis=-1))
        else:
            return unused_outputs[0]

    def __call__(self, inputs, input_conv_1x1):
        self._input_conv_1x1 = input_conv_1x1
        return self._build_block(inputs)


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
        self.combination_methods = combination_methods

    def sample_arch(self, input_shape, n_arches: int=1) -> List:
        arches = []
        for _ in range(n_arches):
            with tf.variable_scope('input'):
                input_ = keras.layers.Input(input_shape)
                input_conv_1x1 = CONV_1x1()(input_)

            # use the input twice because later layers will have the two previous
            # layers' outputs as possible inputs (except after a reduction)
            inputs = [input_, input_]

            # block = self.sample_block()
            # reduce_block = self.sample_reduce_block()
            block = Block.sample_block(self.n_ops_per_node, self.n_nodes_per_block, self.combination_methods)
            reduce_block = Block.sample_block(self.n_ops_per_node, self.n_nodes_per_block, self.combination_methods, stride=2)

            # you can also select the input of the last block as input to a node
            # in the next block (as long as there was no reduce in between)
            last_block_input = input_
            for repeat_idx in range(self.n_block_repeats):
                with tf.variable_scope(f'repeat_{repeat_idx}'):
                    for block_idx in range(self.n_blocks_between_reduce):
                        with tf.variable_scope(f'block_{block_idx}'):
                            block_output = block(inputs, input_conv_1x1)
                        inputs = [last_block_input, block_output]
                        last_block_input = block_output

                    # don't add a reduction block just before the output layer
                    if repeat_idx != self.n_block_repeats - 1:
                        with tf.variable_scope(f'reduce'):
                            block_output = reduce_block(inputs, input_conv_1x1)
                        inputs = [block_output, block_output]
                        last_block_input = block_output

            with tf.variable_scope('output'):
                output = keras.layers.Flatten()(block_output)
                output = keras.layers.Dense(1)(output)

            model = keras.models.Model(input_, output)
            check_blocks([w.name for w in model.trainable_weights], self.n_ops_per_node)
            arches.append(Model(model, block, reduce_block))
        return arches


class WeightsLoader:
    # TODO: do we need to process the names at all?
    # yes... because each block will have something like 'repeat_0/block_0/node_0/Conv5_0/kernel:0'
    # we just want to know node_0/op/input/[kernel or bias]
    def __init__(self, weights_file: str):
        """
        :param weights_file: should end in .npz
        """
        assert weights_file.endswith('.npz')
        self.weights_file = weights_file
        self._load_weights_from_disk()

    def _load_weights_from_disk(self):
        try:
            self.weights = dict(np.load(self.weights_file))
        except FileNotFoundError:
            self.weights = {}

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
    def __init__(self, keras_model: keras.models.Model, block: Block, reduce_block: Block):
        self.keras_model = keras_model
        self.block = block
        self.reduce_block = reduce_block
        self._fitness = None

    def train(self):
        self.keras_model.compile()
        self.keras_model.fit()

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
        block = self.block
        reduce_block = self.reduce_block
        if np.random.randint(2):
            block = block.mutate()
        else:
            reduce_block = reduce_block.mutate()
        # give arch sampler a function to return an arch made from a given set of blocks
        # child = Model(self.arch)
        # child.train()
        return self


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

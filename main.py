# from future import __annotations__
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from tf_layers.tf_utils import tf_init
from evo.main import ArchitectureSampler

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-no', '--n_ops_per_node', type=int, default=2)
    parser.add_argument('-nn', '--n_nodes_per_block', type=int, default=5)
    parser.add_argument('-nb', '--n_block_repeats', type=int, default=3)
    parser.add_argument('-nr', '--n_blocks_between_reduce', type=int, default=2)
    parser.add_argument('-cm', '--combination_methods', nargs='+', default=['add'],
                        help='One or more of {add, mult}; which elementwise operation to use'
                              'when combing the ops for a node (if add and mult are both given,'
                              'one of them will be selected to use for each node).')
    parser.add_argument('-nf', '--n_filters', type=int, default=128,
                        help='The same number of filters is used throughout the network.')
    parser.add_argument('-na', '--n_arches', type=int, help='Number of architectures in population.', required=True)
    parser.add_argument('-ns', '--n_sample', type=int, help='Number of models in a sample.', required=True)
    parser.add_argument('-ng', '--n_generations', type=int, required=True)
    parser.add_argument('-p', '--experiment_path', help='Location to save all experiment data.', required=True)
    parser.add_argument('-ip', '--input_shape', type=int, n_args='+', required=True)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    regularized_evolution = True  # remove oldest arch from sample instead of worst

    # set up the initial population
    arch_sampler = ArchitectureSampler(input_shape=args.input_shape, **dict(args))

    config = tf_init()
    sess = tf.Session(config=config)

    with sess.as_default():
        models = arch_sampler.sample_arch(args.input_shape)

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

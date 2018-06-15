# from future import __annotations__
from argparse import ArgumentParser
from comet_ml import Experiment
import pickle
import os
import numpy as np
import tensorflow as tf
from tf_layers.tf_utils import tf_init
from nn_evolution.architecture import ArchitectureSampler, WeightsLoader
from nn_evolution.utils import save_arches, load_arches
from antibody_design.src.utils import load_seqs, make_splits, encode
import logging


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
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    regularized_evolution = True  # remove oldest arch from sample instead of worst
    n_train_epochs = 3

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        os.mkdir(args.experiment_path)
    except FileExistsError:
        pass

    try:
        with open(f"{os.environ['HOME']}/.comet_key") as f:
            comet_key = f.read().strip()
        exp = Experiment(comet_key, project_name='evo', log_graph=False, auto_metric_logging=False)
    except FileNotFoundError:
        exp = None

    # TODO: make data loading more modular...
    tasks = ['J3/J2']
    seqs = load_seqs(tasks)
    splits = make_splits(seqs.index, seqs[tasks], test_frac=0.1, val_frac=0.1, by_value=False)

    max_seq_len = seqs.index.str.len().max()
    for split in splits.keys():
        if len(splits[split].inputs):
            splits[split].inputs = encode(splits[split].inputs.str.pad(max_seq_len, side='right', fillchar='J'))
    input_shape = splits.train.inputs.shape[1:]

    # set up the initial population
    arch_sampler = ArchitectureSampler(input_shape=input_shape, **vars(args))

    # load existing arches first
    try:
        all_trained_arches = load_arches(f"{args.experiment_path}/all_arches.pkl")

        logging.info(f"Loaded {len(all_trained_arches)} arches.")

        # keep the proper population size at most
        if regularized_evolution:  # the most recent ones were the population when saved
            arches = all_trained_arches[-args.n_arches:]
        else:
            # need to sort by fitness then select the most fit n_arches
            raise NotImplementedError
    except FileNotFoundError:
        arches = []
        all_trained_arches = []

    arches.extend(arch_sampler.sample_arches(args.n_arches - len(arches)))

    weights_file = f'{args.experiment_path}/weights.npz'
    weights_loader = WeightsLoader(weights_file)

    config = tf_init()

    logging.info(f"Training initial population ({args.n_arches} arches).")

    for i in range(len(arches)):
        arch = arches[i]
        if arch._fitness is None:
            arch.train(splits.train.inputs, splits.train.labels, splits.val.inputs, splits.val.labels,
                    weights_loader=weights_loader, sess_config=config, epochs=n_train_epochs)
            if exp:
                exp.log_metric('fitness', arch.fitness, step=-i - 1)
            all_trained_arches.append(arch)
            save_arches(all_trained_arches, f'{args.experiment_path}/all_arches.pkl')
            logging.info(f"Trained initial arch {i}.")

    logging.info(f"Finished initial population. Mutating for {args.n_generations} generations.")

    for generation in range(len(all_trained_arches) - args.n_arches, args.n_generations):
        # select a random sample from the population; mutate the best, remove the oldest or worse
        sample_arches = np.random.choice(arches, size=args.n_sample, replace=False)
        best_arch = max(sample_arches, key=lambda arch: arch.fitness)
        new_arch = best_arch.make_child(arch_sampler)
        new_arch.train(splits.train.inputs, splits.train.labels, splits.val.inputs, splits.val.labels,
                       weights_loader=weights_loader, sess_config=config, epochs=n_train_epochs)

        # the error happens in another thread, so we need to do something in the main thread too
        if new_arch._fitness is None:
            err_arch_fname = f'{args.experiment_path}/err_arch.pkl'
            with open(err_arch_fname, 'wb') as f:
                pickle.dump(new_arch.serialize(), f)
            assert False, f"Error training model! Arch saved to {err_arch_fname}."

        arches.append(new_arch)
        all_trained_arches.append(new_arch)
        save_arches(all_trained_arches, f'{args.experiment_path}/all_arches.pkl')

        if exp:
            exp.log_metric('fitness', new_arch.fitness, step=generation)

        if regularized_evolution:
            worst = max(sample_arches, key=lambda arch: arch.age)
        else:
            worst = min(sample_arches, key=lambda arch: arch.fitness)

        arches.remove(worst)

        mean_fitness = sum(m.fitness for m in arches) / len(arches)
        logging.info(f'Generation {generation}: Removed arch with fitness = {worst.fitness}, '
                     f'added arch with fitness = {new_arch.fitness} (mean fitness = {mean_fitness}).')

        if exp:
            exp.log_metric('mean_fitness', mean_fitness, step=generation)

    with open(f'{args.experiment_path}/best_arch.pkl', 'wb') as f:
        pickle.dump(max(arches, key=lambda arch: arch.fitness).serialize(), f)

import re
import pickle
from typing import List
import nn_evolution.architecture


def check_blocks(model, n_ops_per_node: int, n_nodes_per_block: int) -> None:
    """
    Check that the model is composed only of repetitions of a block and reduce block.

    This doesn't check everything but should work well; it's assumed there will
    be some input/output operations that are excluded from this check, and we also
    don't look at the potential 1x1 convs used to change filter dimensionality.
    We also don't look at the combination method, which should exist once per node.
    This does check that the same nodes in each block apply the same operations to the
    same inputs (same in terms of input index), based on the variable scopes.
    If this isn't the case, `AssertionError` is thrown.
    """
    layer_names = []
    for layer in model.layers:
        try:
            layer_names.append(layer.scope_name)
        except ValueError: # I only get this for "input_1"
            layer_names.append(layer.name)
    layer_names = sorted(layer_names)

    block_names = [n for n in layer_names if 'reduce' not in n]
    reduce_names = [n for n in layer_names if 'reduce' in n]

    node_pattern = 'node_\d+/inp_\d+_\d+/op_\d+/\w+'
    block_names = [re.search(node_pattern, n) for n in block_names]
    reduce_names = [re.search(node_pattern, n) for n in reduce_names]

    block_names = [n.group() for n in block_names if n is not None]
    reduce_names = [n.group() for n in reduce_names if n is not None]

    grouped_blocks = {}
    for name in block_names:
        key = name.split('/')[0]
        if key in grouped_blocks:
            grouped_blocks[key].append(name)
        else:
            grouped_blocks[key] = [name]

    grouped_reduce = {}
    for name in reduce_names:
        key = name.split('/')[0]
        if key in grouped_reduce:
            grouped_reduce[key].append(name)
        else:
            grouped_reduce[key] = [name]

    for names_list in grouped_blocks.values():
        assert len(set(names_list)) == n_ops_per_node, set(names_list)

    for names_list in grouped_reduce.values():
        assert len(set(names_list)) == n_ops_per_node, set(names_list)

    assert len(grouped_blocks) == n_nodes_per_block, grouped_blocks.keys()
    assert len(grouped_reduce) == n_nodes_per_block, grouped_reduce.keys()


def save_arches(arches: list, fname: str) -> None:
    with open(fname, 'wb') as f:
        pickle.dump([arch.serialize() for arch in arches], f)

def load_arches(fname: str) -> list:
    with open(fname, 'rb') as f:
        arches = pickle.load(f)
    arches = [nn_evolution.architecture.Architecture.deserialize(arch) for arch in arches]
    return arches

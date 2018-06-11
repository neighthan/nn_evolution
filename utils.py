import re
from typing import List


def check_blocks(weight_names: List[str], n_ops_per_node: int):
    """This doesn't quite check everything but should work well."""
    weight_names = [w.replace('kernel:0', '').replace('bias:0', '') for w in weight_names if 'input' not in w and 'output' not in w]
    block_names = [w for w in weight_names if 'reduce' not in w]
    reduce_names = [w for w in weight_names if 'reduce' in w]

    node_pattern = 'node_\d+/inp_\d+_\d+/op_\d+/\w+/\w+:\d+'
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

n_ops_per_node="2"
n_nodes_per_block="5"
n_block_repeats="3"
n_blocks_between_reduce="2"
combination_methods="add"
n_filters="128"
n_arches="2"
n_sample="2"
n_generations="2"
experiment_path="/cluster/nhunt/evo_test"
verbose="--verbose" # --verbose if you want this; it's a flag

script_dir="$(dirname "$(realpath "$0")")"

python $script_dir/main.py \
  --n_ops_per_node $n_ops_per_node \
  --n_nodes_per_block $n_nodes_per_block \
  --n_block_repeats $n_block_repeats \
  --n_blocks_between_reduce $n_blocks_between_reduce \
  --combination_methods $combination_methods \
  --n_filters $n_filters \
  --n_arches $n_arches \
  --n_sample $n_sample \
  --n_generations $n_generations \
  --experiment_path $experiment_path \
  $verbose
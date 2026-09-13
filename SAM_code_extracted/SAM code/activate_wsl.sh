# Source this file to activate the Linux environment and use this code directory.
_earthquake_sam_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
source "$HOME/.venvs/earthquake-sam/bin/activate" || return 1
cd -- "$_earthquake_sam_code_dir" || return 1
unset _earthquake_sam_code_dir
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

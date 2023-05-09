set -x 
export HWLOC_COMPONENTS=-gl

CUDA_VISIBLE_DEVICES=$1 mpirun -np $2 python3 tools/run_net.py --config-file=$3 $4
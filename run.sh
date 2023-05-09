set -x 

CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=./:$PYTHONPATH python3 tools/run_net.py --config-file=$2 $3
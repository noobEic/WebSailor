##############Evaluation Parameters################
export MODEL_PATH=$1 # Model path

# Dataset names (strictly match the following names):
# - gaia
# - browsecomp_zh (Full set, 289 Cases)
# - browsecomp_en (Full set, 1266 Cases)
# - xbench-deepsearch
export DATASET=$2 
export OUTPUT_PATH=$3 # Output path for prediction results

export TEMPERATURE=0.6 # LLM generation parameter, fixed at 0.6

apt update
apt install tmux -y
pip install nvitop


echo "==== Starting Original Model vLLM Server (Port 6001)... ===="
CUDA_VISIBLE_DEVICES=0,1 vllm serve $MODEL_PATH --host 0.0.0.0 --tensor-parallel-size 4 --port 6001 &

SUMMARY_SERVER_PID=$!

timeout=3000
start_time=$(date +%s)
server2_ready=false

while true; do
    # Check Local Model
    if ! $server1_ready && curl -s http://localhost:6001/v1/chat/completions > /dev/null; then
        echo -e "\nLocal model (port 6001) is ready!"
        server1_ready=true
    fi
    
    
    # Check if timeout
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout ]; then
        echo -e "\nWarning: Server startup timeout after ${timeout} seconds"
        if ! $server1_ready; then
            echo "vLLM server (port 6001) failed to start"
        fi
        break
    fi
    printf 'Waiting for servers to start .....'
    sleep 10
done

if $server1_ready; then
    echo "Proceeding with the server..."
else



echo "==== Starting inference... ===="
# Activate inference conda environment
export QWEN_DOC_PARSER_USE_IDP=false
export QWEN_IDP_ENABLE_CSI=false
export NLP_WEB_SEARCH_ONLY_CACHE=false
export NLP_WEB_SEARCH_ENABLE_READPAGE=false
export NLP_WEB_SEARCH_ENABLE_SFILTER=false
export QWEN_SEARCH_ENABLE_CSI=false
export SPECIAL_CODE_MODE=false

export MAX_WORKERS=20

python -u run_multi_react_api.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model $MODEL_PATH --temperature $TEMPERATURE 


#####################################
### 4. Start evaluation          ####
#####################################

SUMMARY_PATH="${OUTPUT_PATH}/${DATASET}_summary_api.jsonl"
export MODEL_NAME=$(basename ${MODEL_PATH}) 
PREDICTION_PATH="${OUTPUT_PATH}/${MODEL_NAME}_vllm/${DATASET}"

echo "Evaluating predictions in $PREDICTION_PATH"
python evaluate.py --input_folder ${PREDICTION_PATH} --restore_result_path ${SUMMARY_PATH} --dataset ${DATASET}

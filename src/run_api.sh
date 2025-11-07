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

echo "==== Starting Summary Model vLLM Server (Port 6002)... ===="
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
    --model-path $SUMMARY_MODEL_PATH --host 0.0.0.0 --tp 4 --port 6002 &

SUMMARY_SERVER_PID=$!

timeout=3000
start_time=$(date +%s)
server2_ready=false

while true; do
    # Check Summary Model
    if ! $server2_ready && curl -s http://localhost:6002/v1/chat/completions > /dev/null; then
        echo -e "\nSummary model (port 6002) is ready!"
        server2_ready=true
    fi
    
    # If both servers are ready, exit loop
    if $server2_ready; then
        echo "Both servers are ready for inference!"
        break
    fi
    
    # Check if timeout
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout ]; then
        echo -e "\nWarning: Server startup timeout after ${timeout} seconds"
        if ! $server2_ready; then
            echo "Second server (port 6002) failed to start"
        fi
        break
    fi
    printf 'Waiting for servers to start .....'
    sleep 10
done

if $server2_ready; then
    echo "Proceeding with both servers..."
else
    echo "Proceeding with available servers..."
fi






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

python -u run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model $MODEL_PATH --temperature $TEMPERATURE 


#####################################
### 4. Start evaluation          ####
#####################################

SUMMARY_PATH="${OUTPUT_PATH}/${DATASET}_summary.jsonl"
export MODEL_NAME=$(basename ${MODEL_PATH}) 
PREDICTION_PATH="${OUTPUT_PATH}/${MODEL_NAME}_sglang/${DATASET}"

echo "Evaluating predictions in $PREDICTION_PATH"
python evaluate.py --input_folder ${PREDICTION_PATH} --restore_result_path ${SUMMARY_PATH} --dataset ${DATASET}

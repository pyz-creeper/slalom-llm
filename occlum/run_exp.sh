bash build_instance.sh

cd occlum_instance
OCCLUM_LOG_LEVEL=ERROR SGX_MODE=SIM occlum run /bin/python3 /chatglm2/test.py \
    --model gpt2 --context 128 --batch_size 8

OCCLUM_LOG_LEVEL=ERROR SGX_MODE=SIM occlum run /bin/python3 /chatglm2/test.py \
    --model llama_7b --context 8 --batch_size 1
OCCLUM_LOG_LEVEL=ERROR SGX_MODE=SIM occlum run /bin/python3 /chatglm2/test.py \
    --model bert --context 128 --batch_size 8
cd ../
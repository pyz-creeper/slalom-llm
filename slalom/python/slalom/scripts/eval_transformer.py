from python.slalom.sgxdnn import SGXDNNUtils

import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', type=str,
                        choices=['bert', 'gpt2', 'llama'])

    parser.add_argument('--batch_size', type=int, default=8,
                        help='How many images process at one time.')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='How many images process at one time.')
    parser.add_argument('--max_index', type=int, default=30522)
    args = parser.parse_args()
    return args

def get_random(shape):
    return np.random.random(shape).astype(np.float32).reshape(-1)

def get_json_gpt2(arg):
    def get_block():
        ln1 = {"name": "LayerNorm", "num_feature": 768}
        att = {"name": "SelfAttention", "num_head": 12, "head_dim": 64,
               "sub_layer":[
                   {"name": "Dense", "kernel_size": [768, 768], "activation": "linear"},
                   {"name": "Dense", "kernel_size": [768, 768], "activation": "linear"},
                   {"name": "Dense", "kernel_size": [768, 768], "activation": "linear"},
                   {"name": "Dense", "kernel_size": [768, 768], "activation": "linear"}
               ]}
        ln2 = {"name": "LayerNorm", "num_feature": 768}
        up_proj = {"name": "Dense", "kernel_size": [768, 3072], "activation": "gelu"}
        down_proj = {"name": "Dense", "kernel_size": [3072, 768], "activation": "linear"}
        weights = [get_random((768,)), get_random((768,)), 
                   get_random((768, 768)), get_random((768,)), get_random((768, 768)), get_random((768,)), get_random((768, 768)), get_random((768,)), get_random((768, 768)), get_random((768,)),
                   get_random((768,)), get_random((768,)),
                   get_random((768, 3072)), get_random((3072,)), get_random((3072, 768)), get_random((768,)),]
        return [ln1, att, ln2, up_proj, down_proj], weights
    final_json = {
        "layers": [
            {"name":"Input", "shape":[1,1,1,arg.seq_len]},
            {"name":"Embedding", "max_idx": arg.max_index, "seq_len":arg.seq_len, "num_feature":768}
            ],
        "max_tensor_size": 800000000,
        "shift_w": 8,
        "shift_x": 8,
    }
    final_weight = [get_random((arg.max_index, 768))]
    for i in range(12):
        block_json, block_weight = get_block()
        final_json["layers"].extend(block_json)
        final_weight.extend(block_weight)
    return final_json, final_weight

def get_json_bert(arg):
    def get_block():
        ln1 = {"name": "LayerNorm", "num_feature": 768}
        att = {"name": "SelfAttention", "num_head": 12, "head_dim": 64,
               "sub_layer":[
                   {"name": "Dense", "kernel_size": [768, 768], "activation": "linear"},
                   {"name": "Dense", "kernel_size": [768, 768], "activation": "linear"},
                   {"name": "Dense", "kernel_size": [768, 768], "activation": "linear"},
                   {"name": "Dense", "kernel_size": [768, 768], "activation": "linear"}
               ]}
        ln2 = {"name": "LayerNorm", "num_feature": 768}
        up_proj = {"name": "Dense", "kernel_size": [768, 3072], "activation": "gelu"}
        down_proj = {"name": "Dense", "kernel_size": [3072, 768], "activation": "linear"}
        weights = [get_random((768, 768)), get_random((768,)), get_random((768, 768)), get_random((768,)), get_random((768, 768)), get_random((768,)), get_random((768, 768)), get_random((768,)),
                   get_random((768,)), get_random((768,)),
                   get_random((768, 3072)), get_random((3072,)), get_random((3072, 768)), get_random((768,)),
                   get_random((768,)), get_random((768,)),]
        return [att, ln1, up_proj, down_proj, ln2,], weights
    final_json = {
        "layers": [
            {"name":"Input", "shape":[1,1,1,arg.seq_len]},
            {"name":"Embedding", "max_idx": arg.max_index, "seq_len":arg.seq_len, "num_feature":768}
            ],
        "max_tensor_size": 800000000,
        "shift_w": 8,
        "shift_x": 8,
    }
    final_weight = [get_random((arg.max_index, 768))]
    for i in range(12):
        block_json, block_weight = get_block()
        final_json["layers"].extend(block_json)
        final_weight.extend(block_weight)
    return final_json, final_weight

def get_json_llama(arg):
    def get_block():
        ln1 = {"name": "LayerNorm", "num_feature": 4096}
        attn = {"name": "SelfAttention", "num_head": 32, "head_dim": 128,
               "sub_layer":[
                   {"name": "Dense", "kernel_size": [4096, 4096], "activation": "linear"},
                   {"name": "Dense", "kernel_size": [4096, 4096], "activation": "linear"},
                   {"name": "Dense", "kernel_size": [4096, 4096], "activation": "linear"},
                   {"name": "Dense", "kernel_size": [4096, 4096], "activation": "linear"}
               ]}
        ln2 = {"name": "LayerNorm", "num_feature": 4096}
        mlp = {"name": "llamamlp", 
               "sub_layer":[
                {"name": "Dense", "kernel_size": [4096, 11008], "activation": "linear"},
                {"name": "Dense", "kernel_size": [4096, 11008], "activation": "linear"},
                {"name": "Dense", "kernel_size": [11008, 4096], "activation": "linear"},
               ]}
        weights = [get_random((4096,)), get_random((4096,)),
                   get_random((4096, 4096)), get_random((4096,)), get_random((4096, 4096)), get_random((4096,)),
                   get_random((4096, 4096)), get_random((4096,)), get_random((4096, 4096)), get_random((4096,)),
                   get_random((4096,)), get_random((4096,)),
                   get_random((4096, 11008)), get_random((11008,)), get_random((4096, 11008)), get_random((11008,)), get_random((11008, 4096)), get_random((4096,)),]
        return [ln1,attn, ln2, mlp], weights
    final_json = {
        "layers": [
            {"name":"Input", "shape":[1,1,1,arg.seq_len]},
            {"name":"Embedding", "max_idx": arg.max_index, "seq_len":arg.seq_len, "num_feature":4096}
            ],
        "max_tensor_size": 80000000000,
        "shift_w": 8,
        "shift_x": 8,
    }
    final_weight = [get_random((arg.max_index, 4096))]
    for i in range(32):
        block_json, block_weight = get_block()
        final_json["layers"].extend(block_json)
        final_weight.extend(block_weight)
    return final_json, final_weight

def get_json(arg):
    model_name = arg.model_name
    if model_name == "gpt2":
        return get_json_gpt2(arg)
    elif model_name == "bert":
        return get_json_bert(arg)
    elif model_name == "llama":
        return get_json_llama(arg)

def main():
    args = parse_args()
    sgxutils = SGXDNNUtils(True)
    dtype = np.float32
    model_json, weights = get_json(args)
    sgxutils.load_model(model_json, weights, dtype=dtype, verify=False, verify_preproc=False)
    for i in range(args.batch_size):
        sgxutils.predict_transformer(np.random.randint(0, args.max_index-1, size=(args.seq_len, 1)))


if __name__ == "__main__":
    main()

import argparse
import time
import torch
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer, LlamaModel, LlamaForCausalLM

# Define model functions
def get_model_gpt2():
    return GPT2Model._from_config(GPT2Model.config_class())

def get_model_gpt2_lmhead():
    return GPT2LMHeadModel.from_pretrained("gpt2")

def get_gpt2_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")

def get_model_bert():
    return BertModel._from_config(BertModel.config_class())

def get_bert_tokenizer():
    return BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

def get_llama_7b():
    return LlamaModel._from_config(LlamaModel.config_class())

def get_llama_7b_lmhead():
    return LlamaForCausalLM.from_pretrained("huggyllama/llama-7b")

# Function to get the model based on the model name
def get_model(model_name):
    if model_name == "gpt2":
        return get_model_gpt2()
    elif model_name == "gpt2_lmhead":
        return get_model_gpt2_lmhead()
    elif model_name == "bert":
        return get_model_bert()
    elif model_name == "llama_7b":
        return get_llama_7b()
    elif model_name == "llama_7b_lmhead":
        return get_llama_7b_lmhead()
    else:
        raise ValueError(f"Model {model_name} not supported.")

# Function to run the model inference and measure the average time
def run_inference(model, tokenizer, context, batch_size):
    model.eval()  # Set model to evaluation mode
    inputs = torch.randint(0, 20000, size=(1,context))
    # Move the input tensors to the same device as the model
    device = "cpu"
    model.to(device)

    # Measure time for batch processing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(batch_size):
            model(inputs)
    end_time = time.time()

    # Compute average time per batch
    avg_time_per_batch = (end_time - start_time) / batch_size
    return avg_time_per_batch

# Main function to parse arguments and run the test
def main():
    parser = argparse.ArgumentParser(description="Test transformer models")
    parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., gpt2, bert, llama_7b)')
    parser.add_argument('--context', type=int, required=True, help='Exact number of tokens for inference')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for inference')

    args = parser.parse_args()

    # Load the model and tokenizer
    model = get_model(args.model_name)

    # Run inference and print average time per batch
    avg_time_per_batch = run_inference(model, None, args.context, args.batch_size)
    print(f"Average time per batch: {avg_time_per_batch:.4f} seconds")

if __name__ == "__main__":
    main()

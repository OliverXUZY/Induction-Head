
import os
import requests
import pickle
import json
from datasets import load_dataset, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from .gen_simple import gen_simple_data
import torch


def download_dataset(streaming_samples = 5000, input_file_path = 'data/subset5000.jsonl'):
    dataset = "wikitext"
    ds = load_dataset(
            "Skylion007/openwebtext",
            streaming=True,
            split="train",
        )
    key = "text"

    streaming_samples = 5000


def load_dataset_for_training(tokenizer, block_size=128, verbose = True):
    dataset = "wikitext"
    ds = load_dataset(
            "Skylion007/openwebtext",
            streaming=True,
            split="train",
        )

    train_size = 0.9  # 90% for training, 10% for validation
    train_dataset = ds

    
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)


    tokenized_train = train_dataset.map(tokenize_function, batched=True)


    train_sets = tokenized_train.map(group_texts, batched=True, fn_kwargs={"block_size": block_size})

    if verbose:
        print("see first example: ")
        for example in train_sets.take(1):  # Adjust the number in take() to see more examples
            print(len(example['input_ids']))
            print(example['text'])
            print(tokenizer.decode(example['text']))

    
    return train_sets



def group_texts(examples, block_size = 4):
    print("block_size: ", block_size)

    print(examples.keys())
    print(len(examples))

    
    # Concatenate all texts.
    # text = examples['text'][0]
    # ids = examples['input_ids'][0]
    # print(len(text.split(" ")))
    # print(len(ids))
    # assert False

    examples['text'] = examples['input_ids']
    # for k in examples.keys():
    #     print(k)
    #     print(type(examples[k]))
    #     print(examples[k][:5])
    #     print(type(sum(examples[k], [])))
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def create_dataset(tokenizer, vocab, max_seq_len, sample_size, pattern, verbose = True):
    val = ["Lizard", "koala", "panda", "snake", "cheetah"]
    mapping = {}
    for i in vocab:
        i = i.item()
        # print("vocab: ", i.item())
        mapping[i] = val[i]
    # Generate data
    data = gen_simple_data(vocab, max_seq_len, sample_size, pattern)
    # print(data)
    # print(data.shape)
    
    # Convert to a suitable format, e.g., a list of dictionaries
    formatted_data = {'id': data}

    # Create and return a Dataset object
    raw_dataset = Dataset.from_dict(formatted_data)

    # print(raw_dataset[0])
    

    def id_to_text(example):
        list_of_words = [mapping[i] for i in example['id']]
        example['text'] = " ".join(list_of_words)

        return example
    
    raw_dataset = raw_dataset.map(id_to_text)
    print(raw_dataset[0])
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_len)

    # Tokenize the dataset
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

    if verbose:
        print(tokenized_dataset[0])

    return tokenized_dataset




def main():
    # f = download_dataset()
    # print(f)

    # examples = {
    # 'text': [
    #     [0, 1, 2, 3],          # text 1
    #     [4, 5, 6, 7, 8, 9],    # text 2
    #     [10, 11, 12]           # text 3
    #     ]
    # }

    # print(group_texts(examples))

    
    
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    
    train_sets = load_dataset_for_training(tokenizer = tokenizer)

    # Iterate over the first few examples of the dataset and print them
    for example in train_sets.take(1):  # Adjust the number in take() to see more examples
        print(len(example['input_ids']))
        print(example['text'])
        print(tokenizer.decode(example['text']))
    
    print("============")
    
    

    vocab = torch.arange(5).type(torch.LongTensor)
    

    data = gen_simple_data(vocab, max_seq_len = 128, sample_size = 20, pattern="random")
    

    tokenized_dataset = create_dataset(tokenizer=tokenizer, vocab = vocab, max_seq_len = 60, sample_size = 20, pattern="random")

    print(tokenized_dataset[0])






if __name__ == "__main__":
    main()
import copy
import json
import jsonlines
import io
import logging
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2":(
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_input_llama2": (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} \n{input} [/INST]"
    ),
    "prompt_llama2": "[INST]{instruction}[/INST]"
}

def smart_tokenizer_and_embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def get_mturk_mappings(data_path):
    mturk_ids = []
    all_examples = []

    with jsonlines.open(data_path, 'r') as reader:
        for item in reader:
            mturk_id = item.get("Mturk_id") or item.get("MturkID")
            mturk_ids.append(mturk_id if mturk_id else "unknown")
            all_examples.append(item)

    # Create mapping of Mturk_id to unique integer
    unique_ids = sorted(set(mturk_ids))
    id_to_int = {id_val: i for i, id_val in enumerate(unique_ids)}
    n_users = len(unique_ids)

    # Map each example's Mturk ID to its integer
    mapped_ids = []
    for item in all_examples:
        mturk_id = item.get("Mturk_id") or item.get("MturkID")
        mapped_ids.append(id_to_int.get(mturk_id if mturk_id else "unknown", 0))

    return id_to_int, n_users, mapped_ids



class SupervisedDataset(Dataset):
   """Dataset for supervised fine-tuning."""

   def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, test: bool):
       super(SupervisedDataset, self).__init__()
       logging.warning("Loading data...")
       list_data_dict = jload(data_path)

       if test:
           list_data_dict = list_data_dict[:100]

       # Create Mturk ID mappings
       self.id_to_int, self.n_users, self.mapped_ids = get_mturk_mappings(data_path)

       logging.warning("Formatting inputs...")

       prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_llama2"], PROMPT_DICT["prompt_llama2"]
       sources = [
           prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
           for example in list_data_dict
       ]

       targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

       # Extract user IDs
       self.user_ids = [self.id_to_int.get(example.get("Mturk_id", ""), 0) for example in list_data_dict]

       logging.warning("Tokenizing inputs... This may take some time...")
       data_dict = preprocess(sources, targets, tokenizer)

       self.input_ids = data_dict["input_ids"]
       self.labels = data_dict["labels"]

   def __len__(self):
       return len(self.input_ids)

   def __getitem__(self, i) -> Dict[str, torch.Tensor]:
       return dict(input_ids=self.input_ids[i], labels=self.labels[i], user_id=self.mapped_ids[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        user_ids = torch.tensor(user_ids, dtype=torch.long)

        return dict(
            input_ids=input_ids,
            labels=labels,
            user_ids=user_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, test) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args, test=test)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    n_users = train_dataset.n_users
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator, n_users=n_users)
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
# from transformers import Trainer

from tqdm import tqdm, trange

import os
import deepspeed
import json
import wandb
import time
from transformers import OPTForCausalLM, OPTConfig, AutoTokenizer
from flash_attn.models.opt import remap_state_dict_hf_opt

# os.environ["CUDA_HOME"] = "/opt/apps/cuda/11.7"
# os.environ["LD_LIBRARY_PATH"]= os.environ["CUDA_HOME"] + '/lib64:' + os.environ["LD_LIBRARY_PATH"]
# os.environ["PATH"] = os.environ["CUDA_HOME"] + '/bin:' + os.environ["PATH"]

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
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
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    d_model: Optional[int] = field(default=2048)
    n_heads: Optional[int] = field(default=32)
    n_layers: Optional[int] = field(default=4)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    virtual_batch: Optional[bool] = field(default=False)
    sub_batch_size: Optional[int] = field(default=4)
    cycle_first_step_size: Optional[int] = field(default=600)
    cycle_min_lr: Optional[float] = field(default=1e-5)
    cycle_max_lr: Optional[float] = field(default=1e-3)
    t_max: Optional[int] = field(default=100)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    cache_dir: Optional[str] = field(default='/tmp')
    load_checkpoint: Optional[bool] = field(default=True)



def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
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


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id-1
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).long(),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def ds_init(training_args, model_args, model_conf: str='', ds_conf: str = ''):
    if os.path.exists(model_conf):
        conf = json.load(open(model_conf))
        D_MODEL = conf['D_MODEL']
        N_HEADS = conf['N_HEADS']
        N_LAYERS = conf['N_LAYERS']
    else:
        D_MODEL = model_args.d_model
        N_HEADS = model_args.n_heads
        N_LAYERS = model_args.n_layers
    BATCH_SIZE = training_args.per_device_train_batch_size
    LR = training_args.learning_rate
    MIN_LR = training_args.cycle_min_lr
    MAX_LR = training_args.cycle_max_lr
    os.environ["CUDA_HOME"] = "/opt/apps/cuda/12.0"
    os.environ["LD_LIBRARY_PATH"]= os.environ["CUDA_HOME"] + '/lib64:' + os.environ["LD_LIBRARY_PATH"]
    os.environ["PATH"] = os.environ["CUDA_HOME"] + '/bin:' + os.environ["PATH"]

    if os.path.exists(ds_conf):
        ds_config = ds_conf
    else:
        ds_config = {
            "optimizer": {
                "type": "adamw",
                "params": {
                    "lr": LR,
                }
            },
            # "scheduler": {
            #     "type": "OneCycle",
            #     "params": {
            #         "cycle_first_step_size": 60,
            #         "cycle_min_lr": MIN_LR,
            #         "cycle_max_lr": MAX_LR,
            #     }
            # },
            "bf16": {
                "enabled": True,
            },
            "wandb": {
                "enabled": True,
                "team": "viatage",
                "group": "ds",
                "project": "virtual_batch"
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "nvme",
                    "nvme_path": training_args.cache_dir,
                    "pin_memory": False
                },
                "offload_param": {
                    "device": "nvme",
                    "nvme_path": training_args.cache_dir,
                    "pin_memory": True,
                    "buffer_size": 2e8,
                    "max_in_cpu": 1e8,
                    "buffer_count": 40
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e8,
                "stage3_prefetch_bucket_size": 1e9
            },
            "train_micro_batch_size_per_gpu": BATCH_SIZE,
            "zero_allow_untested_optimizer": True,

        }
    
    # config = GPT2Config(
    #     n_layer=N_LAYERS,
    #     n_embd=D_MODEL,
    #     n_head=N_HEADS,
    #     activation_offload=False,
    #     use_cache=False,
    # )
    config = OPTConfig.from_pretrained(model_args.model_name_or_path)
    config.use_flash_attention = True
    model = OPTForCausalLM(config)
    if training_args.load_checkpoint:
        model = OPTForCausalLM.from_pretrained(model_args.model_name_or_path)
        state_dict = model.state_dict()
        config = model.config
        state_dict_fa = remap_state_dict_hf_opt(state_dict, config)
        model.load_state_dict(state_dict_fa)
    for name, module in model.named_modules():
        module.__name__ = name
        has_child = len(list(module.named_children())) > 0
        if not has_child:
            if hasattr(module, 'weight'):
                module.weight.__name__ = name + ".weight"
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.__name__ = name + ".bias"
    model.gradient_checkpointing_enable()
    model_engine, _, _, _=deepspeed.initialize(
        model = model,
        config = ds_config,
    )
    return model_engine

def build_trace(model_engine, x):
    loss = model_engine(input_ids=x['input_ids'][0].unsqueeze(0).cuda(), 
                            attention_mask=x['attention_mask'][0].unsqueeze(0).cuda(),
                            labels=x['labels'][0].unsqueeze(0).cuda(),
                            return_dict=False)
    model_engine.backward(loss[0])
    model_engine.zero_grad()
    model_engine.step()

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    # )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    model_engine = ds_init(training_args, model_args, model_conf='/home1/09285/zhangyq/work/virtual_batch/gpt2-800M.json')
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model_engine.module,
    )

    


    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    num_train_epochs = int(training_args.num_train_epochs)
    batch_size = training_args.per_device_train_batch_size
    vb_enabled = training_args.virtual_batch 
    vb_sub_size = training_args.sub_batch_size
    dataloader = torch.utils.data.DataLoader(data_module["train_dataset"], batch_size=batch_size, shuffle=True, collate_fn=data_module["data_collator"])
    param_coordinator = None
    if vb_enabled:
        assert batch_size % vb_sub_size == 0, "batch size must be divisible by sub_batch_size"
        param_coordinator = model_engine.optimizer.parameter_offload.get_param_coordinator(training=True)
    # wandb.init(
    #     project="virtual_batch",
    #     name=f"gpt2-800M-vb_{vb_enabled}",
    #     config=training_args,
    # )
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    trace_ready = False

    for epoch in range(num_train_epochs):
        model_engine.train()
        for step, batch in tqdm(enumerate(dataloader), total=dataloader.__len__()):
            with torch.autograd.graph.save_on_cpu():
            # if True:
                if not trace_ready and vb_enabled:
                    build_trace(model_engine, batch)
                    trace_ready = True
                    print("trace ready")

                model_engine.zero_grad()
                batch = {k: v.cuda() for k, v in batch.items()}
                if vb_enabled:
                    if batch['input_ids'].shape[0] != batch_size:
                        continue
                    loss = model_engine(**batch, 
                                        stage=True, 
                                        return_dict=False, 
                                        sub_batch_size=vb_sub_size,
                                        param_coordinator=param_coordinator)[0]
                else:
                    outputs = model_engine(**batch)
                    loss = outputs[0]
                model_engine.backward(loss)
                model_engine.step()
                # wandb.log({"loss": loss.item()/(batch_size//vb_sub_size) if vb_enabled else loss.item(),
                #             "learning_rate": lr_scheduler.get_last_lr()[0],
                # })
                if step % 100 == 0:
                    print(f"epoch: {epoch}, step: {step}, loss: {loss.item()}")

    # return
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    # trainer.train()
    # trainer.save_state()
    # trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

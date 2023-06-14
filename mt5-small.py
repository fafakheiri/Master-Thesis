from transformers import  MT5ForConditionalGeneration, T5Tokenizer
import torch
import yaml
import os
import mlflow
import pandas as pd
import csv
from munch import Munch
from transformers import (set_seed,
                          Seq2SeqTrainingArguments,
                          EarlyStoppingCallback,
                          Trainer,
                          Seq2SeqTrainer,
                          logging,
                          DataCollatorForSeq2Seq
                         )
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Dict
import matplotlib.pyplot as plt
import evaluate
from classes.main import Main
from classes.splitter import Splitter
from classes.args import Args
from classes.eval import Eval
from transformers import EarlyStoppingCallback
import evaluate
import torch
import numpy as np

# Matplot
global x, y
x = []
y = []
# Loading the configuration file into cfg
with open("path_to_the_config_file", "r") as file:
    cfg = yaml.safe_load(file)
# Converting dictionary to object
cfg = Munch(cfg)
# set some params
set_seed(cfg.params["seed"])

Splitter()
# MLflow setup
os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow["exp_name"]
os.environ["MLFLOW_FLATTEN_PARAMS"] = cfg.mlflow["params"]

PROJECT_NAME = cfg.mlflow["exp_name"] + cfg.mlflow["params"]

tokenizer = T5Tokenizer.from_pretrained(cfg.MT["checkpoint"],
src_lang=cfg.MT["src"],
tgt_lang=cfg.MT["tgt"],
do_lower_case=cfg.params["lower_case"],
normalization=cfg.params["normalization"])

raw_datasets = load_dataset("csv", sep='\t', quoting=csv.QUOTE_NONE, data_files={
"train": [cfg.MT["train_path"]],
"dev":   [cfg.MT["dev_path"]],
"test":  [cfg.MT["test_path"]]})

print("HF raw dataset", raw_datasets)
print(raw_datasets["train"][0])
print("special tokens:", tokenizer.special_tokens_map)

def tokenize_function(examples):
    return tokenizer(examples["src"], text_target=examples["tgt"], truncation=True, max_length=cfg.params["max_len"])

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

print("Tokenized dataset: ", tokenized_datasets)

tokenized_datasets = tokenized_datasets.remove_columns(['src', 'tgt'])

tokenized_datasets.set_format("torch")

print(tokenized_datasets)

print("dataset's columns:", tokenized_datasets["train"].column_names)
print("example:", tokenized_datasets['train'][11])
print("decoding the example:", tokenizer.decode(tokenized_datasets["train"]['input_ids'][100]))
print("decoding the example:", tokenizer.decode(tokenized_datasets["train"]['labels'][100]))

print("*** Data Collator ***")

def data_collator(tokenizer):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    return data_collator

data_collator = data_collator(tokenizer)
print("Data Collator: ", data_collator)

def data_loader(tokenized_datasets, data_collator):

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=cfg.params["train_bs"], collate_fn=data_collator
    )
    dev_dataloader = DataLoader(
        tokenized_datasets["dev"], batch_size=cfg.params["dev_bs"], collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=cfg.params["test_bs"], collate_fn=data_collator
    )
    return [train_dataloader, dev_dataloader, test_dataloader]

train_dataloader, dev_dataloader, test_dataloader = data_loader(tokenized_datasets, data_collator)

model = MT5ForConditionalGeneration.from_pretrained(cfg.MT["checkpoint"])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print("You're using:", device)

# Training the model
print("*** Training the model ***")
#Training Arguments
training_args= Seq2SeqTrainingArguments(cfg.mlflow["exp_name"] + cfg.mlflow["params"],
per_device_train_batch_size= cfg.params["device_train_bs"],
per_device_eval_batch_size= cfg.params["device_eval_bs"],
num_train_epochs= cfg.params["epoch"],
logging_steps= cfg.params["logging_step"],
save_total_limit= cfg.params["save"],
save_steps= cfg.params["save_steps"],
weight_decay= cfg.params["weight_decay"],
warmup_steps= cfg.params["warmup_steps"],
seed= cfg.params["seed"],
fp16= cfg.params["fp16"],
push_to_hub= False,
learning_rate= float(cfg.params["lr"]),
evaluation_strategy= cfg.params["evaluation"],
eval_steps= cfg.params["eval_steps"], # Evaluation and Save happens every n steps
load_best_model_at_end= cfg.params["best_model"],
metric_for_best_model= cfg.params["metric_best_model"],
do_train=cfg.MT["do_train"],
do_eval=cfg.MT["do_eval"],
do_predict=cfg.MT["do_predict"],
optim=cfg.params["optim"],
predict_with_generate=cfg.MT["predict_with_generate"],
report_to="none",
)

class MySeq2SeqTrainer(Seq2SeqTrainer):
    def log(self, logs: Dict[str, float]) -> None:
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)

        f = open(PROJECT_NAME + ".txt", "a")
        f.write(str(logs) + '\n' )
        f.close()

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metric(eval_preds):
    preds, labels = eval_preds
    print(len(preds))
    preds = preds[0] if isinstance(preds, tuple) else preds
    print(len(preds))
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    print(pred_str)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print(label_str)

    # Some simple post-processing
    pred_str, label_str = postprocess_text(pred_str, label_str)

    sacrebleu = evaluate.load("sacrebleu")
    ter = evaluate.load("ter")

    result = sacrebleu.compute(predictions=pred_str, references=label_str)
    result2 = ter.compute(predictions=pred_str, references=label_str, ignore_punct=True,)

    result = {"bleu": result["score"]}
    result["TER_score"] = result2["score"]

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    print(list(result.keys()))

    return result

trainer = MySeq2SeqTrainer(
model,
args=training_args,
train_dataset=tokenized_datasets["train"],
eval_dataset=tokenized_datasets["dev"],
data_collator=data_collator,
tokenizer=tokenizer,
# preprocess_logits_for_metrics=preprocess_logits_for_metrics,
callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.params["early_stop"])],
compute_metrics=compute_metric)

# Disable tokenizer_parallelism error
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # Creating an instance (named run) from the main class
    run = Main()
    # Split the text file into train, dev and test sets
    Splitter() 
    # Call load_dataset
    raw_datasets = run.load_dataset()
    # Apply tokenization function to the sample
    tokenized_datasets = raw_datasets.map(
        run.tokenize_function, batched=True)
    # Remove the header (src and tgt) from the input
    tokenized_datasets = tokenized_datasets.remove_columns(['src', 'tgt'])
    # Set the format: torch
    tokenized_datasets.set_format("torch")
    # Call the data collator
    data_collator = run.data_collator(run.tokenizer)
    # Call the data loader
    train_dataloader, dev_dataloader, test_dataloader = run.data_loader(
        tokenized_datasets, run.data_collator)
    # Intialize the model
    model = run.model
    # Load the model in GPU (if available)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    print("You're using:", device)
    # Call the training arguments
    training_args = Args().training_args
    # Define the trainer (derived from MySeq2SeqTrainer)
    trainer = MySeq2SeqTrainer(
        model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=data_collator,
        tokenizer=run.tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=run.cfg.params["early_stop"])],
        compute_metrics=compute_metric
    )
    # Train
    if run.cfg.params["do_train"]==True:
        trainer.train()

    print(trainer.predict(tokenized_datasets["test"]))
    print("# of model's parameters: ", model.num_parameters())
    
if __name__ == '__main__':
    main()
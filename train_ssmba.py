from datasets import load_dataset
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import hydra
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast
import os
from datetime import date
import logging
from omegaconf import DictConfig
from hydra import slurm_utils
import torch.nn as nn
import torch

class KLTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = F.log_softmax(outputs[0], dim=-1, dtype=torch.float32)
        return F.kl_div(logits, labels)

class WeightedTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, 15], dtype=torch.float32).cuda())

    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        return self.loss_fn(outputs[0], labels)


@hydra.main(config_path='/h/nng/conf/biossmba/config.yaml', strict=False)
def train(cfg: DictConfig):
    log = logging.getLogger(__name__)

    base_dir = '/h/nng/data/readmit/mimic/ssmba_' + str(cfg.noise)

    input_dataset = load_dataset('text', data_files={'train': base_dir + '/train', 'valid': base_dir + '/valid', 'test': base_dir + '/test'})
    if cfg.label == 'soft':
        trainer_cls = KLTrainer
        label_dataset = load_dataset('text', data_files={'train': base_dir + '/train_label_soft', 'valid': base_dir + '/valid_label_soft', 'test': base_dir + '/test_label_soft'})

    elif cfg.label == 'hard':
        trainer_cls = WeightedTrainer
        label_dataset = load_dataset('text', data_files={'train': base_dir + '/train_label_hard', 'valid': base_dir + '/valid_label_hard', 'test': base_dir + '/test_label_hard'})

    elif cfg.label == 'pres':
        trainer_cls = WeightedTrainer
        label_dataset = load_dataset('text', data_files={'train': base_dir + '/train_label', 'valid': base_dir + '/valid_label', 'test': base_dir + '/test_label'})

    print(input_dataset)
    print(label_dataset)

    tokenizer = DistilBertTokenizerFast.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    #tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def encode_inputs(examples):
         return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    def encode_soft_labels(example):
        probs = example['text'].strip().split()
        example['0'] = float(probs[0])
        example['1'] = float(probs[1])
        return example

    if cfg.label == 'soft':
        label_dataset = label_dataset.map(encode_soft_labels)
    input_dataset = input_dataset.map(encode_inputs, batched=True)

    # training model on tokenized and split data

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, inputs, labels, label_type):
            self.inputs = inputs
            self.labels = labels
            self.label_type = label_type

        def __getitem__(self, idx):
            item = {key: torch.tensor(val) for key, val in self.inputs[idx].items() if key != 'text'}
            if self.label_type == 'soft':
                item['labels'] = torch.tensor([self.labels[idx]['0'], self.labels[idx]['1']])
            else:
                item['labels'] = torch.tensor(int(self.labels[idx]['text']))

            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(input_dataset['train'], label_dataset['train'], cfg.label)
    val_dataset = Dataset(input_dataset['valid'], label_dataset['valid'], cfg.label)
    test_dataset = Dataset(input_dataset['test'], label_dataset['test'], cfg.label)

    j_dir = slurm_utils.get_j_dir(cfg)
    o_dir = os.path.join(j_dir, os.environ['SLURM_JOB_ID'])
    log_dir = os.path.join(o_dir, 'logs', os.environ['SLURM_JOB_ID'])
    os.makedirs(log_dir, exist_ok=True)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model = model.cuda()

    def compute_metrics(pred):
        if cfg.label == 'soft':
            labels = pred.label_ids.argmax(-1)
        else:
            labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        roc = roc_auc_score(labels, pred.predictions[:,-1])
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auroc': roc,
        }

    training_args = TrainingArguments(
        output_dir=o_dir,          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=10800,                # number of warmup steps for learning rate scheduler
        weight_decay=0.1,               # strength of weight decay
        logging_dir=log_dir,
        logging_steps=100,
        evaluation_strategy='steps',
        learning_rate=2e-5,
        fp16=True,
        save_total_limit=5,
        eval_steps=1000,
        save_steps=1000,
    )

    training_args._n_gpu = 8

    trainer = trainer_cls(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
    )

    # check for a previous checkpoint
    found = False
    for job in sorted(os.listdir(j_dir))[::-1]:
        if found:
            break
        for f in sorted(os.listdir(os.path.join(j_dir, job)))[::-1]:
            if 'checkpoint' in f:
                found = True
                model_dir = os.path.join(j_dir, job, f)
                break


    print(found, flush=True)
    if found:
        print(model_dir, flush=True)
    if found:
        trainer.train(model_dir)
    else:
        trainer.train()

if __name__ == "__main__":
    train()

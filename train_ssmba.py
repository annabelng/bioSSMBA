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

class KLTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = F.log_softmax(outputs[0], dim=-1, dtype=torch.float32)
        return F.kl_div(logits, labels)


@hydra.main(config_path='/h/nng/conf/biossmba/config.yaml', strict=False)
def train(cfg: DictConfig):
    log = logging.getLogger(__name__)

    base_dir = '/h/nng/data/readmit/mimic/ssmba_0.1'

    input_dataset = load_dataset('text', data_files={'train': base_dir + '/train', 'valid': base_dir + '/valid', 'test': base_dir + '/test'})
    label_dataset = load_dataset('text', data_files={'train': base_dir + '/train_label', 'valid': base_dir + '/valid_label', 'test': base_dir + '/test_label'})

    print(input_dataset)
    print(label_dataset)

    tokenizer = DistilBertTokenizerFast.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    #tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def encode_inputs(examples):
         return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    def encode_labels(example):
        probs = example['text'].strip().split()
        example['0'] = float(probs[0])
        example['1'] = float(probs[1])
        return example

    label_dataset = label_dataset.map(encode_labels)
    input_dataset = input_dataset.map(encode_inputs, batched=True)

    # training model on tokenized and split data

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val) for key, val in self.inputs[idx].items() if key != 'text'}
            item['labels'] = torch.tensor([self.labels[idx]['0'], self.labels[idx]['1']])
            #item['labels'] = torch.tensor(int(self.labels[idx]['text']))

            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(input_dataset['train'], label_dataset['train'])
    val_dataset = Dataset(input_dataset['valid'], label_dataset['valid'])
    test_dataset = Dataset(input_dataset['test'], label_dataset['test'])

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model = model.cuda()

    def compute_metrics(pred):
        labels = pred.label_ids.argmax(-1)
        #labels = pred.label_ids
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

    j_dir = slurm_utils.get_j_dir(cfg)
    o_dir = os.path.join(j_dir, os.environ['SLURM_JOB_ID'])
    log_dir = os.path.join(o_dir, 'logs', os.environ['SLURM_JOB_ID'])
    os.makedirs(log_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=o_dir,          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=12500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.1,               # strength of weight decay
        logging_dir=log_dir,
        logging_steps=100,
        evaluation_strategy='steps',
        learning_rate=4e-5,
        fp16=True,
        save_total_limit=5,
        eval_steps=2000,
        save_steps=2000,
    )

    training_args._n_gpu = 4

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    train()

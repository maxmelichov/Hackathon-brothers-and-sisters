import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import accuracy_score
from datasets import DatasetDict


def get_model_args(dataset_dict: DatasetDict):
    batch_size = 8
    logging_steps = len(dataset_dict["train"]) // batch_size
    model_name = f"distilbert-base-cased-finetuned"
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=2,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    log_level="error",
                                    optim='adamw_torch'
                                    )
    return training_args

def train_model(model: torch.nn.Module, dataset_dict: DatasetDict, training_args: TrainingArguments, tokenizer: AutoTokenizer):
    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=dataset_dict["train"],
                    eval_dataset=dataset_dict["validation"],
                    tokenizer=tokenizer)
    trainer.train()
    return trainer

def get_predictions(trainer: Trainer, dataset_dict: DatasetDict):
    predictions = trainer.predict(dataset_dict["test"])
    predictions.predictions.argmax(axis=-1)
    return predictions


def get_accuracy(preds):
  predictions = preds.predictions.argmax(axis=-1)
  labels = preds.label_ids
  accuracy = accuracy_score(labels, predictions)
  return {'accuracy': accuracy}


import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from datasets import DatasetDict


def get_model_args() -> TrainingArguments:
    model_name = f"distilbert-base-cased-finetuned"
    training_args = TrainingArguments(
                    output_dir = model_name,          # output directory to where save model checkpoint
                    evaluation_strategy = "steps",    # evaluate each `logging_steps` steps
                    overwrite_output_dir = True,      
                    num_train_epochs = 5,            # number of training epochs, feel free to tweak
                    per_device_train_batch_size = 128, # the training batch size, put it as high as your GPU memory fits
                    gradient_accumulation_steps = 8,  # accumulating the gradients before updating the weights
                    per_device_eval_batch_size = 128,  # evaluation batch size
                    logging_steps  =1000,             # evaluate, log and save model checkpoints every 1000 step
                    save_steps = 1000,
                    load_best_model_at_end = True,  # whether to load the best model (in terms of loss) at the end of training
                    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
                    auto_find_batch_size = True
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


def get_score(preds):
    predictions = preds.predictions.argmax(axis=-1)
    labels = preds.label_ids
    accuracy = accuracy_score(labels, predictions)

    f1 = f1_score(labels, predictions)
    recall = recall_score(labels, predictions)
    precision = precision_score(labels, predictions)
    return {'accuracy': accuracy, 'f1': f1, "recall": recall,"precision":precision }


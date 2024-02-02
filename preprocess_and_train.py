import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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

def load_data(dir_file: str) -> pd.DataFrame:
    df = pd.read_csv(dir_file)
    return df

def combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df1, df2], ignore_index=True)
    return df

def get_tokenizer(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    outputs = tokenizer(df["text"].tolist(), padding = True, truncation = True)
    df["input_ids"] = outputs["input_ids"]
    df["attention_mask"] = outputs["attention_mask"]
    return df, tokenizer

def load_model(model_name: str) -> torch.nn.Module:
    num_labels = 2
    model = (AutoModelForSequenceClassification
            .from_pretrained(model_name, num_labels = num_labels)
            .to(DEVICE))
    return model

def split_data(df: pd.DataFrame) -> DatasetDict:
    df_train, df_temp = train_test_split(df, test_size = 0.3, random_state = 42)
    df_val, df_test = train_test_split(df_temp, test_size = 0.5, random_state = 42)

    df_train.reset_index(drop = True, inplace = True)
    df_val.reset_index(drop = True, inplace = True)
    df_test.reset_index(drop = True, inplace = True)

    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    return dataset_dict


if "__main__" == __name__:
    path = os.path.join(os.getcwd(),"data")
    Suicide_Detection = load_data(os.path.join(path,"Suicide_Detection.csv"))
    Suicide_Detection.drop("Unnamed: 0", axis = 1, inplace = True)
    Suicide_Detection["class"] = Suicide_Detection["class"].replace("suicide", 1)
    Suicide_Detection["class"] = Suicide_Detection["class"].replace("non-suicide", 0)
    Suicide_Detection.rename(columns={"class": "label"}, inplace=True)
    depression_dataset_reddit_cleaned = load_data(os.path.join(path,"depression_dataset_reddit_cleaned.csv"))
    depression_dataset_reddit_cleaned.rename(columns={"clean_text": "text", "is_depression": "label"}, inplace = True)

    df = combine_dataframes(Suicide_Detection,depression_dataset_reddit_cleaned)
    df.reset_index(inplace = True, drop = True)

    df, tokenizer = get_tokenizer(df, "distilbert-base-cased")
    
    print("Starting Model load and creating dataset_dict... Please wait.")
    model = load_model("distilbert-base-cased")
    dataset_dict = split_data(df)

    training_args = get_model_args()
    trainer = train_model(model, dataset_dict, training_args, tokenizer)
    predictor = get_predictions(trainer, dataset_dict)
    print(get_score(predictor))
    trainer.save_model()

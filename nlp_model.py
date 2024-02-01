import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import re
import openai
import config
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


openai.api_key=config.key
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

def load_data(dir_file:str) -> pd.DataFrame:
    df = pd.read_csv(dir_file)
    return df

def comine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df1, df2], ignore_index=True)
    return df

def get_tokenizer(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenize all texts at once
    outputs = tokenizer(df["text"].tolist(), padding=True, truncation=True)
    # Add tokenized outputs to dataframe
    df["input_ids"] = outputs["input_ids"]
    df["attention_mask"] = outputs["attention_mask"]
    return df, tokenizer

def load_model(model_name: str) -> torch.nn.Module:
    num_labels = 2
    model = (AutoModelForSequenceClassification
            .from_pretrained(model_name, num_labels=num_labels)
            .to(DEVICE))
    return model

def split_data(df: pd.DataFrame) -> DatasetDict:
    
    # Assuming df is your DataFrame

    # Split the DataFrame into train and temp (which will be split again)
    df_train, df_temp = train_test_split(df, test_size=0.3)
    # Split the temp DataFrame into validation and test
    df_val, df_test = train_test_split(df_temp, test_size=0.5)

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # Convert the DataFrames to datasets
    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)

    # Combine the datasets into a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    return dataset_dict

def get_model_args():
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

if "__main__" == __name__:
    Suicide_Detection = load_data(r"data\Suicide_Detection.csv")
    Suicide_Detection.drop("Unnamed: 0", axis = 1, inplace = True)
    Suicide_Detection["class"] = Suicide_Detection["class"].replace("suicide",1)
    Suicide_Detection["class"] = Suicide_Detection["class"].replace("non-suicide",0)
    Suicide_Detection.rename(columns={"class": "label"}, inplace=True)
    print(1)
    depression_dataset_reddit_cleaned = load_data(r"data\depression_dataset_reddit_cleaned.csv")
    depression_dataset_reddit_cleaned.rename(columns={"clean_text": "text", "is_depression": "label"}, inplace=True)
    print(2)
    df = comine_dataframes(Suicide_Detection,depression_dataset_reddit_cleaned)
    df.reset_index(inplace=True, drop=True)
    print(3)
    df, tokenizer = get_tokenizer(df, "distilbert-base-cased")
    print(4)
    model = load_model("distilbert-base-cased")
    dataset_dict = split_data(df)
    print(5)
    training_args = get_model_args()
    trainer = train_model(model, dataset_dict, training_args, tokenizer)
    print(2)
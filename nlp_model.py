import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import openai
import config

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
openai.api_key = config.key

def load_data(dir_file: str) -> pd.DataFrame:
    df = pd.read_csv(dir_file)
    return df

def combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df1, df2], ignore_index=True)
    return df

def get_tokenizer(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    outputs = tokenizer(df["text"].tolist(), padding=True, truncation=True)
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
    df_train, df_temp = train_test_split(df, test_size=0.3)
    df_val, df_test = train_test_split(df_temp, test_size=0.5)

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    return dataset_dict

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

if "__main__" == __name__:
    Suicide_Detection = load_data(r"data\Suicide_Detection.csv")
    Suicide_Detection.drop("Unnamed: 0", axis = 1, inplace = True)
    Suicide_Detection["class"] = Suicide_Detection["class"].replace("suicide",1)
    Suicide_Detection["class"] = Suicide_Detection["class"].replace("non-suicide",0)
    Suicide_Detection.rename(columns={"class": "label"}, inplace=True)
    depression_dataset_reddit_cleaned = load_data(r"data\depression_dataset_reddit_cleaned.csv")
    depression_dataset_reddit_cleaned.rename(columns={"clean_text": "text", "is_depression": "label"}, inplace=True)

    df = combine_dataframes(Suicide_Detection,depression_dataset_reddit_cleaned)
    df.reset_index(inplace=True, drop=True)

    df, tokenizer = get_tokenizer(df, "distilbert-base-cased")

    model = load_model("distilbert-base-cased")
    dataset_dict = split_data(df)

    training_args = get_model_args(dataset_dict)
    trainer = train_model(model, dataset_dict, training_args, tokenizer)

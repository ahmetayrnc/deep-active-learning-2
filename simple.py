from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset, DatasetDict
import torch
import numpy as np
import pandas as pd
import dataiku
from my_utils import save_df


def run_experiment(use_full_dataset=True, sampling="random"):
    # Define the metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro",
            zero_division=0,
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # Load the dataset
    dataset = load_dataset("silicone", "swda")
    dataset = dataset.rename_column("Label", "labels")
    dataset = dataset.rename_column("Utterance", "text")

    # If not using the full dataset, take only 0.1% of it
    if not use_full_dataset:
        dataset = dataset.shuffle(seed=42)  # For consistent results, set a seed
        dataset = DatasetDict(
            {
                split: split_dataset.select(range(int(len(split_dataset) * 0.001)))
                for split, split_dataset in dataset.items()
            }
        )

    # Load the tokenizer and the model
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=46
    )

    # Tokenize the dataset
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    dataset = dataset.map(tokenize, batched=True)
    train_dataset = dataset["train"]

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        disable_tqdm=True,
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=dataset["test"],
    )

    # We start by training the model on 1% of the training data.
    initial_percentage = 0.01
    end_percentage = 0.2
    n_initial = int(initial_percentage * len(train_dataset))
    labeled_pool, unlabeled_pool = torch.utils.data.random_split(
        train_dataset, [n_initial, len(train_dataset) - n_initial]
    )

    # initial training
    trainer.train_dataset = labeled_pool
    trainer.train()

    # Calculate rounds and batch sizes for active learning
    n_rounds = 40
    n_queries = int((len(train_dataset) * end_percentage) // n_rounds)

    results = []
    for round in range(n_rounds):
        # Query the unlabeled pool
        if sampling == "random":
            query_indices = np.random.choice(
                len(unlabeled_pool), n_queries, replace=False
            )
        else:
            print("querying")
            preds = trainer.predict(unlabeled_pool)
            uncertainty_scores = 1 - np.max(preds.predictions, axis=1)
            query_indices = np.argpartition(-uncertainty_scores, n_queries)[:n_queries]

        # move queried instances from unlabeled to labeled pool
        labeled_pool += torch.utils.data.Subset(unlabeled_pool, query_indices)
        unlabeled_pool = torch.utils.data.Subset(
            unlabeled_pool,
            np.delete(np.arange(len(unlabeled_pool)), query_indices),
        )

        # Retrain the model with selected instances
        print("training")
        trainer.train_dataset = labeled_pool
        trainer.train()

        # Evaluate the model
        print("evaluating")
        eval_result = trainer.evaluate()
        eval_result.update({"round": round, "strategy": sampling})
        results.append(eval_result)

        # print(f"Round {round+1} comparison:")
        # pprint(eval_result)

    # Save the results
    results = pd.DataFrame(results)
    results = results.drop(
        columns=[
            "epoch",
            "eval_loss",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
        ]
    )
    results = results.rename(
        columns={
            "eval_accuracy": "accuracy",
            "eval_f1": "f1",
            "eval_precision": "precision",
            "eval_recall": "recall",
        }
    )
    # print(results)

    return results


def run_experiments(use_full_dataset=True):
    results = []
    for sampling in ["random", "uncertainty"]:
        results.append(run_experiment(use_full_dataset, sampling))
    results = pd.concat(results)
    return results


results = run_experiments(False)

# save results
results_folder = dataiku.Folder("KFC4Ufdr")
file_name = f"no_context_al_results.csv"
save_df(results_folder, file_name, results)

import argparse
from pprint import pprint
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)


def main(args):
    model_name = args["model"]
    model: PreTrainedModel = AutoModel.from_pretrained(
        model_name,
        num_labels=46,
    )

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-cased",
        choices=[
            "distilbert-base-cased",
            "distilbert-base-uncased-finetuned-sst-2-english",
        ],
        help="model",
    )

    args = parser.parse_args()
    args_dict = vars(args)
    pprint(args_dict)

    main(args_dict)

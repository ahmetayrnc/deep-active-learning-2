import argparse
import numpy as np
import torch
from data import Data
from utils import get_dataset, get_handler, get_net, get_strategy
from pprint import pprint
import os
import pandas as pd
from transformers import logging as transformers_logging
from SequentialSentenceClassifier import SequentialSentenceClassifier
from utils import default_params


def main(args: dict) -> pd.DataFrame:
    print("[INFO] Running experiment with the following arguments:")
    pprint(args)

    # set environment variable to disable parallelism in tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # disable transformers warnings
    transformers_logging.set_verbosity_error()

    # fix random seed
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.backends.cudnn.enabled = False

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Running experiments on: {device}")

    # load dataset
    print("Loading dataset...")
    train, test = get_dataset(args["dataset_name"])
    print(f"Dataset loaded.")

    number_of_samples = 6
    train = train[0][:number_of_samples], train[1][:number_of_samples]
    handler = get_handler(args["dataset_name"])
    dataset = Data(train, test, handler)
    print(f"Number of samples to train on: {number_of_samples}")

    # load network and strategy
    print("Loading network...")
    if "params" not in args:
        args["params"] = default_params
    net = get_net(args["dataset_name"], device, 1, args["params"])  # load network
    # turn the model to train mode
    net.model = SequentialSentenceClassifier(default_params[args["dataset_name"]])
    net.model.train()
    print(f"Network loaded.")

    print("Loading strategy...")
    strategy = get_strategy(args["strategy_name"])(dataset, net)  # load strategy
    print(f"Strategy loaded.")

    # start experiment
    dataset.initialize_labels(0)

    query_idxs = strategy.query(1)
    print(f"query indices: {query_idxs}")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SWDA",
        choices=["SWDA", "DYDA", "KPN"],
        help="dataset",
    )
    parser.add_argument(
        "--strategy_name",
        type=str,
        default="RandomSampling",
        choices=[
            "RandomSampling",
            "MaxTurnUncertainty",
            "MinTurnUncertainty",
            "AverageTurnUncertainty",
            "MedianTurnUncertainty",
        ],
        help="query strategy",
    )

    args = parser.parse_args()
    args_dict = vars(args)

    main(args_dict)

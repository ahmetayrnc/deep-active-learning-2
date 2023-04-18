import argparse
import numpy as np
import torch
from data import Data
from nets import HierarchicalDialogueActClassifier
from utils import get_dataset, get_handler, get_net, get_strategy
from pprint import pprint
import os
import pandas as pd


def main(args: dict) -> pd.DataFrame:
    # set environment variable to disable parallelism in tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    subset = 20
    train = np.array(train[0][:subset], dtype=object), np.array(
        train[1][:subset], dtype=object
    )
    test = np.array(test[0], dtype=object), np.array(test[1], dtype=object)
    handler = get_handler(args["dataset_name"])
    dataset = Data(train, test, handler)
    print(f"Dataset loaded.")

    # load network and strategy
    print("Loading network and strategy...")
    net = get_net(args["dataset_name"], device, args["n_epoch"])  # load network

    net.model: HierarchicalDialogueActClassifier = HierarchicalDialogueActClassifier(
        net.params["model_name"], net.params["n_labels"]
    )
    net.model.train()

    strategy = get_strategy(args["strategy_name"])(dataset, net)  # load strategy
    print(f"Network and strategy loaded.")

    # start experiment
    dataset.initialize_labels(args["n_init_labeled"])
    print(f"number of labeled pool: {args['n_init_labeled']}")
    print(f"number of unlabeled pool: {dataset.n_pool-args['n_init_labeled']}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    query_idxs = strategy.query(args["n_query"])
    print(f"query indices: {query_idxs}")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--epoch", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument(
        "--n_init_labeled",
        type=int,
        default=10,
        help="number of init labeled samples",
    )
    parser.add_argument(
        "--n_query", type=int, default=1, help="number of queries per round"
    )
    parser.add_argument("--n_round", type=int, default=10, help="number of rounds")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SWDA",
        choices=["SWDA"],
        help="dataset",
    )
    parser.add_argument(
        "--strategy_name",
        type=str,
        default="RandomSampling",
        choices=[
            "RandomSampling",
            "MaxTurnUncertainty",
            "MarginSampling",
            "EntropySampling",
            "LeastConfidenceDropout",
            "MarginSamplingDropout",
            "EntropySamplingDropout",
            "KMeansSampling",
            "KCenterGreedy",
            "BALDDropout",
            "AdversarialBIM",
            "AdversarialDeepFool",
        ],
        help="query strategy",
    )

    args = parser.parse_args()
    args_dict = vars(args)
    pprint(args_dict)

    main(args_dict)

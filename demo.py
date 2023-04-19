import argparse
import numpy as np
import torch
from data import Data
from utils import get_dataset, get_handler, get_net, get_strategy
from pprint import pprint
import os
import pandas as pd
from transformers import logging as transformers_logging


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
    handler = get_handler(args["dataset_name"])
    dataset = Data(train, test, handler)
    print(f"Dataset loaded.")

    # load network and strategy
    print("Loading network and strategy...")
    net = get_net(args["dataset_name"], device, args["n_epoch"])  # load network
    strategy = get_strategy(args["strategy_name"])(dataset, net)  # load strategy
    print(f"Network and strategy loaded.")

    # start experiment
    dataset.initialize_labels(args["n_init_labeled"])
    print(f"number of labeled pool: {args['n_init_labeled']}")
    print(f"number of unlabeled pool: {dataset.n_pool-args['n_init_labeled']}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    # initialize results
    results = []
    experiment_name = f"dataset_name:{args['dataset_name']}+n_init_labeled:{args['n_init_labeled']}+n_query:{args['n_query']}+n_round:{args['n_round']}+epoch:{args['n_epoch']}+seed:{args['seed']}+strategy_name:{args['strategy_name']}"

    # round 0 accuracy
    print("Round 0")
    strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    metrics = dataset.cal_test_metrics(preds)
    print(f"Round 0 testing metrics: {metrics}")

    # collect information about the round
    round_summary = {
        "experiment": experiment_name,
        "round": 0,
    }
    round_summary.update(args)
    round_summary.update(metrics)
    results.append(round_summary)

    # start active learning
    for rd in range(1, args["n_round"] + 1):
        print(f"Round {rd}")

        # query
        print("Querying...")
        query_idxs = strategy.query(args["n_query"])

        # update labels
        print("Updating labels...")
        strategy.update(query_idxs)

        print("Training...")
        strategy.train()

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        metrics = dataset.cal_test_metrics(preds)
        print(f"Round {rd} testing metrics: {metrics}")

        # collect information about the round
        round_summary = {
            "experiment": experiment_name,
            "round": rd,
        }
        round_summary.update(args)
        round_summary.update(metrics)
        results.append(round_summary)

    results = pd.DataFrame(results)
    return results


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--n_epoch", type=int, default=1, help="number of epochs to train"
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
        help="dataset to use",
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
        help="query strategy to use",
    )

    args = parser.parse_args()
    args_dict = vars(args)
    pprint(args_dict)

    main(args_dict)

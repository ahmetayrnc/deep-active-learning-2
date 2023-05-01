import argparse
import datetime
import numpy as np
import torch
from data import Data
from utils import get_dataset, get_handler, get_net, get_strategy
from pprint import pprint
import os
import time
import pandas as pd
from transformers import logging as transformers_logging


def main(args: dict) -> pd.DataFrame:
    print("[INFO] Running experiment with the following arguments:")
    pprint(args)

    # get current date time
    current_time = datetime.datetime.now()
    startdate = current_time.strftime("%Y-%m-%d %H:%M:%S")

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
    net = get_net(name=args["dataset_name"], device=device, n_epoch=args["n_epoch"])
    strategy = get_strategy(
        args["strategy_name"],
        dataset,
        net,
        args["agg"],
        args["clipping"],
    )
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
    training_loss = strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    metrics = dataset.cal_test_metrics(preds)
    print(f"Round 0 testing metrics: {metrics}")

    # collect information about the round
    round_summary = {
        "round": 0,
        "training_loss": training_loss,
        "labeled_data": dataset.get_labeled_data()[0].shape[0],
        "query_elapsed_time": 0,
        "elapsed_time": 0,
    }
    round_summary.update(args)
    round_summary.update(metrics)
    results.append(round_summary)

    # start active learning
    start_time = time.time()
    for rd in range(1, args["n_round"] + 1):
        print(f"Round {rd}")

        # query
        print("Querying...")
        query_start_time = time.time()
        query_idxs = strategy.query(args["n_query"])
        query_elapsed_time = time.time() - query_start_time

        # update labels
        print("Updating labels...")
        strategy.update(query_idxs)

        print("Training...")
        training_loss = strategy.train()

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        metrics = dataset.cal_test_metrics(preds)
        print(f"Round {rd} testing metrics: {metrics}")

        # collect information about the round
        labeled_data_size = dataset.get_labeled_data()[0].shape[0]
        cumulative_elapsed_time = time.time() - start_time
        round_summary = {
            "round": rd,
            "training_loss": training_loss,
            "labeled_data": labeled_data_size,
            "query_elapsed_time": query_elapsed_time,
            "elapsed_time": cumulative_elapsed_time,
        }
        round_summary.update(args)
        round_summary.update(metrics)
        results.append(round_summary)

    results = pd.DataFrame(results)
    results = results.assign(startdate=startdate, experiment=experiment_name)

    return results


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SWDA",
        choices=["SWDA", "DYDA", "KPN"],
        help="dataset to use",
    )
    parser.add_argument(
        "--strategy_name",
        type=str,
        default="RandomSampling",
        choices=[
            "RandomSampling",
            "TurnUncertainty",
            "TurnEntropy",
            "TurnMargin",
        ],
        help="query strategy to use",
    )
    parser.add_argument(
        "--agg",
        type=str,
        default="max",
        choices=["min", "max", "mean", "median"],
        help="aggregation method to use",
    )
    parser.add_argument(
        "--clipping",
        type=int,
        default=0,
        help="how much to clip the output of the network",
    )
    parser.add_argument(
        "--n_epoch", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument(
        "--n_init_labeled",
        type=int,
        default=10,
        help="number of init labeled samples",
    )
    parser.add_argument("--n_round", type=int, default=10, help="number of rounds")
    parser.add_argument(
        "--n_query", type=int, default=1, help="number of queries per round"
    )

    args = parser.parse_args()
    args_dict = vars(args)

    main(args_dict)

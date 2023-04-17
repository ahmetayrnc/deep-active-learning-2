import argparse
import numpy as np
import torch
from data import Data
from utils import get_dataset, get_handler, get_net, get_strategy
from pprint import pprint
import os
import pandas as pd


def main(args):
    # set environment variable to disable parallelism in tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Running experiments on: {device}")

    # load dataset
    print("Loading dataset...")
    train, test = get_dataset(args.dataset_name)
    handler = get_handler(args.dataset_name)
    dataset = Data(train, test, handler)
    print(f"Dataset loaded.")

    # load network and strategy
    print("Loading network and strategy...")
    net = get_net(args.dataset_name, device)  # load network
    strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy
    print(f"Network and strategy loaded.")

    # start experiment
    dataset.initialize_labels(args.n_init_labeled)
    print(f"number of labeled pool: {args.n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    # initialize results
    results = []
    experiment_name = f"dataset_name:{args.dataset_name}+n_init_labeled:{args.n_init_labeled}+n_query:{args.n_query}+n_round:{args.n_round}+seed:{args.seed}+strategy_name:{args.strategy_name}"

    # round 0 accuracy
    print("Round 0")
    strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    metrics = dataset.cal_test_metrics(preds)
    print(f"Round 0 testing metrics: {metrics}")

    # information about the current round
    round_summary = {
        "experiment": experiment_name,
        "dataset_name": args.dataset_name,
        "n_init_labeled": args.n_init_labeled,
        "n_query": args.n_query,
        "n_round": args.n_round,
        "seed": args.seed,
        "strategy_name": args.strategy_name,
        "round": 0,
    }

    # add the metrics to the round information
    round_summary.update(metrics)

    # add it to the results
    results.append(round_summary)

    # start active learning
    for rd in range(1, args.n_round + 1):
        print(f"Round {rd}")

        # query
        print("Querying...")
        query_idxs = strategy.query(args.n_query)

        # update labels
        print("Updating labels...")
        strategy.update(query_idxs)

        print("Training...")
        strategy.train()

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        metrics = dataset.cal_test_metrics(preds)
        print(f"Round {rd} testing metrics: {metrics}")

        # information about the current round
        round_summary = {
            "experiment": experiment_name,
            "dataset_name": args.dataset_name,
            "n_init_labeled": args.n_init_labeled,
            "n_query": args.n_query,
            "n_round": args.n_round,
            "seed": args.seed,
            "strategy_name": args.strategy_name,
            "round": rd,
        }

        # add the metrics to the round information
        round_summary.update(metrics)

        # add it to the results
        results.append(round_summary)

    results = pd.DataFrame(results)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
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
            "LeastConfidence",
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
    pprint(vars(args))
    print()

    main(args)

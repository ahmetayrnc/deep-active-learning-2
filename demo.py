import argparse
import numpy as np
import torch
from data import Data
from utils import get_dataset, get_handler, get_net, get_strategy
from pprint import pprint
import os

# set environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument(
    "--n_init_labeled", type=int, default=10000, help="number of init labeled samples"
)
parser.add_argument(
    "--n_query", type=int, default=1000, help="number of queries per round"
)
parser.add_argument("--n_round", type=int, default=10, help="number of rounds")
parser.add_argument(
    "--dataset_name",
    type=str,
    default="MNIST",
    choices=["MNIST", "SWDA"],
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

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load dataset
X_tr, Y_tr, X_te, Y_te, A_tr, A_te = dataset = get_dataset(args.dataset_name)
handler = get_handler(args.dataset_name)
dataset = Data(X_tr, Y_tr, X_te, Y_te, A_tr, A_te, handler)

# load network and strategy
net = get_net(args.dataset_name, device)  # load network
strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

# start experiment
dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

# round 0 accuracy
print("Round 0")
strategy.train()
preds = strategy.predict(dataset.get_test_data())
print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")

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
    print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")

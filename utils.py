from typing import Type
from numpy import ndarray
from data import get_SWDA
from nets import Net, SWDA_Net
from handlers import Conversation_Handler
from query_strategies.strategy import Strategy
from torch.utils.data import Dataset
from query_strategies import (
    RandomSampling,
    LeastConfidence,
    MarginSampling,
    EntropySampling,
    LeastConfidenceDropout,
    MarginSamplingDropout,
    EntropySamplingDropout,
    KMeansSampling,
    KCenterGreedy,
    BALDDropout,
    AdversarialBIM,
    AdversarialDeepFool,
)

params = {
    "SWDA": {
        "n_epoch": 1,
        "train_args": {"batch_size": 128, "num_workers": 0},
        "test_args": {"batch_size": 1000, "num_workers": 0},
        "optimizer_args": {"lr": 0.05, "momentum": 0.3},
    },
}


def get_handler(name: str) -> Type[Dataset]:
    if name == "SWDA":
        return Conversation_Handler
    else:
        raise NotImplementedError


def get_dataset(
    name: str,
) -> "tuple[dict[str, ndarray], dict[str, ndarray], dict[str, ndarray]]":
    if name == "SWDA":
        return get_SWDA()
    else:
        raise NotImplementedError


def get_net(name: str, device: str) -> Net:
    if name == "SWDA":
        return Net(SWDA_Net, params[name], device)
    else:
        raise NotImplementedError


def get_params(name: str) -> "dict[str, object]":
    return params[name]


def get_strategy(name: str) -> Type[Strategy]:
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "LeastConfidenceDropout":
        return LeastConfidenceDropout
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "KMeansSampling":
        return KMeansSampling
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialBIM":
        return AdversarialBIM
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool
    else:
        raise NotImplementedError

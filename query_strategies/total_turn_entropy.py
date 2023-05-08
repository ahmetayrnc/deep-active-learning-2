import numpy as np
from .strategy import Strategy
from scipy.stats import entropy


class TotalTurnEntropy(Strategy):
    def __init__(self, dataset, net, ignore_top_k_percent: int = 5):
        super(TotalTurnEntropy, self).__init__(dataset, net)
        self.ignore_top_k_percent = ignore_top_k_percent

    def query(self, n):
        # Get the indices and data of the unlabeled dialogues from the dataset
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # Predict the probabilities for each unlabeled dialogue using the neural network model
        probs = self.predict_prob(unlabeled_data)

        # Calculate the entropies for each turn in the dialogues
        entropies = [entropy(turn_probs.T) for turn_probs in probs]

        # Apply the ignore_top_k_percent option before aggregation
        if self.ignore_top_k_percent > 0:
            all_entropies = np.concatenate(entropies)
            threshold = np.percentile(
                all_entropies,
                100 - self.ignore_top_k_percent,
            )
            entropies = [
                np.where(entropy < threshold, entropy, -np.inf) for entropy in entropies
            ]

        # Aggregate the entropies for each dialogue based on the specified method
        aggregated_entropies = [
            np.sum(dialogue_entropies) for dialogue_entropies in entropies
        ]

        # Select the indices of the top n dialogues with the highest aggregated entropies
        selected_dialogue_indices = np.argpartition(aggregated_entropies, -n)[-n:]

        # Retrieve the original indices of the selected dialogues in the unlabeled dataset
        selected_dialogue_idxs = [
            unlabeled_idxs[idx] for idx in selected_dialogue_indices
        ]

        # Return the selected dialogue indices
        return selected_dialogue_idxs

import numpy as np
from .strategy import Strategy


class TurnUncertainty(Strategy):
    def __init__(self, dataset, net, ignore_top_k_percent: int = 5):
        super(TurnUncertainty, self).__init__(dataset, net)
        self.ignore_top_k_percent = ignore_top_k_percent

    def query(self, n):
        # Get the indices and data of the unlabeled dialogues from the dataset
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # Predict the probabilities for each unlabeled dialogue using the neural network model
        probs = self.predict_prob(unlabeled_data)

        # Calculate the uncertainties for each turn in the dialogues
        uncertainties = [1 - np.max(turn_probs, axis=1) for turn_probs in probs]

        # Apply the ignore_top_k_percent option before aggregation
        if self.ignore_top_k_percent > 0:
            all_uncertainties = np.concatenate(uncertainties)
            threshold = np.percentile(
                all_uncertainties,
                100 - self.ignore_top_k_percent,
            )
            uncertainties = [
                np.where(uncertainty < threshold, uncertainty, -np.inf)
                for uncertainty in uncertainties
            ]

        # Compute the aggregated uncertainty for each dialogue
        aggreagated_uncertainties = [
            np.mean(dialogue_uncertainties) for dialogue_uncertainties in uncertainties
        ]

        # Select the indices of the top n dialogues with the highest aggregated uncertainties
        selected_dialogue_indices = np.argpartition(aggreagated_uncertainties, -n)[-n:]

        # Retrieve the original indices of the selected dialogues in the unlabeled dataset
        selected_dialogue_idxs = [
            unlabeled_idxs[idx] for idx in selected_dialogue_indices
        ]

        # Return the selected dialogue indices
        return selected_dialogue_idxs

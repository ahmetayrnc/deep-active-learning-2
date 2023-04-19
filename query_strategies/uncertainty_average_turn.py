import numpy as np
from .strategy import Strategy


class AverageTurnUncertainty(Strategy):
    def __init__(self, dataset, net):
        # Call the parent class constructor with the provided dataset and neural network model
        super(AverageTurnUncertainty, self).__init__(dataset, net)

    def query(self, n):
        # Get the indices and data of the unlabeled dialogues from the dataset
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # Predict the probabilities for each unlabeled dialogue using the neural network model
        probs = self.predict_prob(unlabeled_data)

        # Calculate the uncertainties for each turn in the dialogues
        uncertainties = [1 - np.max(turn_probs, axis=1) for turn_probs in probs]

        # Compute the average uncertainty for each dialogue
        avg_uncertainties = [
            np.mean(dialogue_uncertainties) for dialogue_uncertainties in uncertainties
        ]

        # Select the indices of the top n dialogues with the highest average uncertainties
        selected_dialogue_indices = np.argpartition(avg_uncertainties, -n)[-n:]

        # Retrieve the original indices of the selected dialogues in the unlabeled dataset
        selected_dialogue_idxs = [
            unlabeled_idxs[idx] for idx in selected_dialogue_indices
        ]

        # Return the selected dialogue indices
        return selected_dialogue_idxs

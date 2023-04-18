import numpy as np
from .strategy import Strategy


# class LeastConfidence(Strategy):
#     def __init__(self, dataset, net):
#         super(LeastConfidence, self).__init__(dataset, net)

#     def query(self, n):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         probs = self.predict_prob(unlabeled_data)
#         # List[ndarray[num_turns, num_classes] of length num_dialogues
#         # select the dialogue with the highest average uncertainty
#         uncertainties = probs.max(1)[0]
#         return unlabeled_idxs[uncertainties.sort()[1][:n]]


class MaxTurnUncertainty(Strategy):
    def __init__(self, dataset, net):
        super(MaxTurnUncertainty, self).__init__(dataset, net)

    def query(self, n):
        print("querying...")
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        print("unlabeled_idxs: ", unlabeled_idxs)

        # List[ndarray[num_turns, num_classes] of length num_dialogues
        # predict the probabilities for each dialogue
        print("predicting...")
        probs = self.predict_prob(unlabeled_data)
        print("probs: ", probs)

        # Calculate uncertainties (1 - max_probability) for each turn prediction
        print("calculating uncertainties...")
        uncertainties = [1 - np.max(turn_probs, axis=1) for turn_probs in probs]
        print("uncertainties: ", uncertainties)

        # Find the maximum uncertainty for each dialogue
        print("finding max uncertainties...")
        max_uncertainties = [
            np.max(dialogue_uncertainties) for dialogue_uncertainties in uncertainties
        ]
        print("max uncertainties: ", max_uncertainties)

        # Select the top n dialogues with the highest maximum uncertainties
        print("selecting top n dialogues...")
        selected_dialogue_indices = np.argpartition(max_uncertainties, -n)[-n:]
        print("selected_dialogue_indices: ", selected_dialogue_indices)

        print("returning selected_dialogue_idxs...")
        selected_dialogue_idxs = [
            unlabeled_idxs[idx] for idx in selected_dialogue_indices
        ]
        print("selected_dialogue_idxs: ", selected_dialogue_idxs)

        return selected_dialogue_idxs

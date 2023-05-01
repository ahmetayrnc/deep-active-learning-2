import numpy as np
from .strategy import Strategy


class TurnMargin(Strategy):
    def __init__(self, dataset, net, aggregation="max", ignore_top_k_percent: int = 5):
        super(TurnMargin, self).__init__(dataset, net)
        self.ignore_top_k_percent = ignore_top_k_percent

        # Aggregate the entropies for each dialogue based on the specified method
        if aggregation == "max":
            self.agg = np.max
        elif aggregation == "min":
            self.agg = np.min
        elif aggregation == "mean":
            self.agg = np.mean
        elif aggregation == "median":
            self.agg = np.median
        else:
            self.agg = np.max

    def query(self, n):
        # Get the indices and data of the unlabeled dialogues from the dataset
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # Predict the probabilities for each unlabeled dialogue using the neural network model
        probs = self.predict_prob(unlabeled_data)

        # Calculate the margin for each turn in the dialogues
        margins = [self.margin(turn_probs) for turn_probs in probs]

        # Apply the ignore_top_k_percent option before aggregation
        if self.ignore_top_k_percent > 0:
            all_margins = np.concatenate(margins)
            threshold = np.percentile(
                all_margins,
                100 - self.ignore_top_k_percent,
            )
            margins = [
                np.where(margin < threshold, margin, -np.inf) for margin in margins
            ]

        # Aggregate the margins for each dialogue based on the specified method
        aggregated_margins = [
            self.agg(dialogue_margins) for dialogue_margins in margins
        ]

        # Select the indices of the top n dialogues with the highest aggregated margins
        selected_dialogue_indices = np.argpartition(aggregated_margins, -n)[-n:]

        # Retrieve the original indices of the selected dialogues in the unlabeled dataset
        selected_dialogue_idxs = [
            unlabeled_idxs[idx] for idx in selected_dialogue_indices
        ]

        # Return the selected dialogue indices
        return selected_dialogue_idxs

    @staticmethod
    def margin(probabilities):
        sorted_probs = np.sort(probabilities, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        return margin

import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from data import Data
from nets import Net
from ..strategy import Strategy


# Define the submodular function for informativeness and diversity
def submodular_function(
    dialogues: List[int],
    uncertainties: List[float],
    embeddings: List[np.ndarray],
    alpha: float = 0.5,
) -> float:
    # Calculate the sum of uncertainties for the selected dialogues
    informativeness = np.sum([uncertainties[idx] for idx in dialogues])

    # Calculate the cosine similarity matrix for the selected dialogue embeddings
    diversity_matrix = cosine_similarity(
        np.array([embeddings[idx] for idx in dialogues]).reshape(-1, 1)
    )

    # Calculate diversity by subtracting the sum of the similarity matrix and adding the number of dialogues
    # This will remove the diagonal elements (similarity to themselves)
    diversity = -np.sum(diversity_matrix) + len(dialogues)

    # Return the weighted sum of informativeness and diversity
    return alpha * informativeness + (1 - alpha) * diversity


class Submodular(Strategy):
    def __init__(self, dataset: Data, net: Net):
        # Call the parent class constructor with the provided dataset and neural network model
        super(Submodular, self).__init__(dataset, net)

    def query(self, n: int) -> List[int]:
        # Get the indices and data of the unlabeled dialogues from the dataset
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # Predict the probabilities for each unlabeled dialogue using the neural network model
        probs, embeds = self.predict_prob_with_embeddings(unlabeled_data)

        # Get the embeddings for each unlabeled dialogue
        avg_embeddings = [
            np.mean(dialogue_embeddings) for dialogue_embeddings in embeds
        ]

        # Calculate the uncertainties for each turn in the dialogues
        uncertainties = [1 - np.max(turn_probs, axis=1) for turn_probs in probs]

        # Compute the average uncertainty for each dialogue
        avg_uncertainties = [
            np.mean(dialogue_uncertainties) for dialogue_uncertainties in uncertainties
        ]

        # Select the indices of the dialogues that maximize the submodular function
        selected_dialogue_idxs = self.select_dialogues_submodular(
            unlabeled_idxs, avg_uncertainties, avg_embeddings, n
        )

        # Return the selected dialogue indices
        return selected_dialogue_idxs

    @staticmethod
    def select_dialogues_submodular(
        unlabeled_idxs: List[int],
        uncertainties: List[float],
        embeddings: List[np.ndarray],
        n: int,
        alpha: float = 0.5,
    ) -> List[int]:
        selected_dialogue_idxs = []

        # Iterate n times to select n dialogues
        for _ in range(n):
            best_dialogue_idx = -1
            best_function_value = -np.inf

            # Iterate through the unlabeled dialogues to find the one that maximizes the submodular function
            for idx in range(len(unlabeled_idxs)):
                if idx not in selected_dialogue_idxs:
                    # Create a temporary list of selected dialogues including the current dialogue
                    current_dialogue_idxs = selected_dialogue_idxs + [idx]

                    # Calculate the submodular function value for the current selection
                    current_function_value = submodular_function(
                        current_dialogue_idxs,
                        uncertainties,
                        embeddings,
                        alpha,
                    )

                    # Update the best dialogue index and function value if the current value is better
                    if current_function_value > best_function_value:
                        best_function_value = current_function_value
                        best_dialogue_idx = idx

            # Add the best dialogue index to the selected dialogues
            selected_dialogue_idxs.append(best_dialogue_idx)

        # Return the original indices of the selected dialogues in the unlabeled dataset
        return [unlabeled_idxs[idx] for idx in selected_dialogue_idxs]

import torch
import torch.nn as nn
from typing import List
from transformers import AutoModel, AutoTokenizer


class HierarchicalDialogueActClassifier(nn.Module):
    def __init__(
        self, pretrained_model_name: str, num_classes: int, max_turn_length: int = 64
    ):
        super(HierarchicalDialogueActClassifier, self).__init__()
        self.max_turn_length = max_turn_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, use_fast=True
        )
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.dialogue_transformer = AutoModel.from_pretrained(pretrained_model_name)

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

        # move everything to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.dialogue_transformer.to(self.device)
        self.classifier.to(self.device)

    def forward(self, dialogues: List[List[str]]):
        dialogue_turns = dialogues[0]

        # Tokenize and move the encoding data to the device
        turn_encodings = [
            self.tokenizer(
                turn,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_turn_length,
            ).to(self.device)
            for turn in dialogue_turns
        ]

        # Obtain turn embeddings
        turn_outputs = [self.encoder(**encoding) for encoding in turn_encodings]
        turn_hidden_states = [
            output.last_hidden_state[:, 0, :] for output in turn_outputs
        ]
        turn_embeddings = torch.stack(turn_hidden_states)

        # Permute the dimensions of the turn_embeddings tensor
        turn_embeddings = turn_embeddings.permute(1, 0, 2)

        # Pass the turn embeddings through the dialogue-level transformer
        dialogue_outputs = self.dialogue_transformer(inputs_embeds=turn_embeddings)
        dialogue_hidden_states = dialogue_outputs.last_hidden_state

        # Use the hidden state of each turn for classification
        logits = self.classifier(dialogue_hidden_states)

        # dialogue_hidden_states: hidden states
        return logits, dialogue_hidden_states

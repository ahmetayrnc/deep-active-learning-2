import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple


class SequentialSentenceClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_classes: int):
        super(SequentialSentenceClassifier, self).__init__()

        # Load a pretrained model and its tokenizer
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, use_fast=True
        )

        # Define a linear classifier to map the hidden states to the desired number of classes
        self.classifier = nn.Linear(
            self.pretrained_model.config.hidden_size, num_classes
        )

        # Move everything to the appropriate device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model.to(self.device)
        self.classifier.to(self.device)

    def forward(self, dialogues: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        all_logits = []
        all_embeddings = []

        # Define a helper function to process chunks of dialogue
        def process_chunk(chunk: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
            # Combine the dialogue sentences using the separator token
            sep_token = self.tokenizer.sep_token
            text = sep_token.join(chunk)

            # Tokenize the combined text and create an attention mask
            tokens = self.tokenizer.encode(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            mask = tokens.ne(self.tokenizer.pad_token_id).float()

            # Get the hidden states of the pretrained model
            outputs = self.pretrained_model(tokens, attention_mask=mask)
            last_hidden_state = outputs.last_hidden_state

            # Find the indices of separator tokens in the tokenized input
            sep_indices = (
                (tokens.squeeze() == self.tokenizer.sep_token_id).nonzero().squeeze()
            )

            # Extract the embeddings for each sentence
            embeddings = last_hidden_state[0, sep_indices]

            # Apply the classifier to obtain logits
            logits = self.classifier(embeddings)

            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
                embeddings = embeddings.unsqueeze(0)

            return logits, embeddings

        # Process each dialogue in the input dialogues
        for dialogue in dialogues:
            dialogue_logits = []
            dialogue_embeddings = []

            # Split the dialogue into smaller chunks and process each chunk
            chunk_size = 20  # Adjust this based on your specific use case
            chunks = [
                dialogue[i : i + chunk_size]
                for i in range(0, len(dialogue), chunk_size)
            ]

            for chunk in chunks:
                logits, embeddings = process_chunk(chunk)
                # logits = logits.unsqueeze(0)
                # print(f"chuck logits shape: {logits.shape}")
                dialogue_logits.append(logits)
                dialogue_embeddings.append(embeddings)

            # Concatenate the logits and embeddings for each dialogue
            # print(dialogue_logits)
            dialogue_logits = torch.cat(dialogue_logits)
            dialogue_embeddings = torch.cat(dialogue_embeddings)
            all_logits.append(dialogue_logits)
            all_embeddings.append(dialogue_embeddings)

        # Stack the logits and embeddings from all dialogues
        logits = torch.stack(all_logits)
        embeddings = torch.stack(all_embeddings)

        return logits, embeddings

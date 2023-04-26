import torch
import torch.nn as nn
from typing import List, Tuple
from transformers import AutoModel, AutoTokenizer
from nets import DatasetArgs


class SequentialSentenceClassifier(nn.Module):
    def __init__(self, params: DatasetArgs):
        super(SequentialSentenceClassifier, self).__init__()
        self.dataset_params = params

        # Load a pretrained model and its tokenizer
        self.pretrained_model = AutoModel.from_pretrained(params["model_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            params["model_name"], use_fast=True
        )

        # Define a classifier to map the hidden states to the desired number of classes
        self.classifier = nn.Linear(
            self.pretrained_model.config.hidden_size, params["n_labels"]
        )

        self.dialogue_length = self.pretrained_model.config.max_position_embeddings

        # Move everything to the appropriate device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model.to(self.device)
        self.classifier.to(self.device)

    def forward(self, dialogues: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        all_logits = []
        all_embeddings = []

        # Define a helper function to process chunks of dialogue
        def process_chunk(chunk: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
            # Truncate each sentence to a maximum of turn_length characters
            truncated_chunk = [
                sentence[: self.dataset_params["turn_length"]] for sentence in chunk
            ]

            # Combine the dialogue sentences using the separator token
            sep_token_id = self.tokenizer.sep_token_id
            sep_token = self.tokenizer.sep_token
            text = sep_token.join(truncated_chunk) + sep_token

            # Tokenize the combined text
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="longest",
                add_special_tokens=False,
                return_attention_mask=True,
            ).to(self.device)

            # Get the hidden states of the pretrained model
            outputs = self.pretrained_model(**tokens)
            last_hidden_state = outputs.last_hidden_state

            # Find the indices of separator tokens in the tokenized input
            sep_indices = (
                (tokens["input_ids"].squeeze() == sep_token_id).nonzero().squeeze()
            )

            # Extract the embeddings for each sentence
            embeddings = last_hidden_state[0, sep_indices]

            # Apply the classifier to obtain logits
            logits = self.classifier(embeddings)

            # if chunk consists of a single sentence, add the sentence dimension
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
                embeddings = embeddings.unsqueeze(0)

            return logits, embeddings

        # Process each dialogue in the input dialogues
        max_len_batch = max(len(dialogue) for dialogue in dialogues)
        for dialogue in dialogues:
            dialogue_logits = []
            dialogue_embeddings = []

            # Split the dialogue into smaller chunks and process each chunk
            # chunk size = 6 mean, 6 turns per chunk
            chunk_size = int(
                (self.dialogue_length * 4) / self.dataset_params["turn_length"]
            )
            chunks = [
                dialogue[i : i + chunk_size]
                for i in range(0, len(dialogue), chunk_size)
            ]

            for chunk in chunks:
                logits, embeddings = process_chunk(chunk)
                dialogue_logits.append(logits)
                dialogue_embeddings.append(embeddings)

            # Concatenate the logits and embeddings for each dialogue
            dialogue_logits = torch.cat(dialogue_logits)
            dialogue_embeddings = torch.cat(dialogue_embeddings)

            # Pad the logits and embeddings to the same length, in order to have a orderly batch
            if dialogue_logits.size(0) < max_len_batch:
                pad_len = max_len_batch - dialogue_logits.size(0)
                pad_logits = torch.zeros(pad_len, dialogue_logits.size(1)).to(
                    self.device
                )
                pad_embeddings = torch.zeros(pad_len, dialogue_embeddings.size(1)).to(
                    self.device
                )
                dialogue_logits = torch.cat([dialogue_logits, pad_logits], dim=0)
                dialogue_embeddings = torch.cat(
                    [dialogue_embeddings, pad_embeddings], dim=0
                )

            # Append the logits and embeddings of the dialogue
            all_logits.append(dialogue_logits)
            all_embeddings.append(dialogue_embeddings)

        # Stack the logits and embeddings from all dialogues
        logits = torch.stack(all_logits)
        embeddings = torch.stack(all_embeddings)

        return logits, embeddings

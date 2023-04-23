import torch
import torch.nn as nn
from typing import List, Tuple
from transformers import AutoModel, AutoTokenizer


class MLP(nn.Module):
    def __init__(
        self, input_size: int, num_classes: int, hidden_sizes: List[int] = None
    ):
        super(MLP, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# self.classifier = MLP(self.pretrained_model.config.hidden_size, num_classes)


class SequentialSentenceClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_classes: int):
        super(SequentialSentenceClassifier, self).__init__()

        # Load a pretrained model and its tokenizer
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, use_fast=True
        )

        # Define a classifier to map the hidden states to the desired number of classes
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
            # Truncate each sentence to a maximum of 320 characters
            truncated_chunk = [sentence[:320] for sentence in chunk]

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
                max_length=4096,
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
        max_len = max(len(dialogue) for dialogue in dialogues)
        for dialogue in dialogues:
            dialogue_logits = []
            dialogue_embeddings = []

            # Split the dialogue into smaller chunks and process each chunk
            chunk_size = 50  # Adjust this based on your specific use case
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

            # Pad the logits and embeddings to the same length
            if dialogue_logits.size(0) < max_len:
                pad_len = max_len - dialogue_logits.size(0)
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

            all_logits.append(dialogue_logits)
            all_embeddings.append(dialogue_embeddings)

        # Stack the logits and embeddings from all dialogues
        logits = torch.stack(all_logits)
        embeddings = torch.stack(all_embeddings)

        return logits, embeddings

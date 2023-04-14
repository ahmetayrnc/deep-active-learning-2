from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

model_name = "distilbert-base-cased"
model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=46,
)

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
)

model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)

from transformers import AutoModelForSequenceClassification, PreTrainedModel

model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-cased",
    num_labels=46,
)

model.save_pretrained("distilbert-base-cased")

from transformers import AutoModelForSequenceClassification, PreTrainedModel

model_name = "distilbert-base-cased"
model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=46,
)

model.save_pretrained(model_name)

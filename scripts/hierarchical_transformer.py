# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from sklearn.metrics import classification_report
# import numpy as np

# # Params
# pretrained_model_name = "distilbert-base-cased"
# # pretrained_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# num_classes = 46
# batch_size = 1
# num_epochs = 1

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = HierarchicalDialogueActClassifier(pretrained_model_name, num_classes)
# # model.to(device)
# print(f'device: {device}')

# train_data = DialogueDataset(train)
# test_data = DialogueDataset(test)

# train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=string_collator)
# test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=string_collator)

# # Define the loss function
# loss_function = nn.CrossEntropyLoss(ignore_index=-1)

# # Define the optimizer
# optimizer = AdamW(model.parameters(), lr=1e-5)

# # Add the accumulation steps parameter
# accumulation_steps = 4

# # Training loop
# for epoch in range(num_epochs):
#     epoch_loss = 0.0
#     accumulation_step = 0
#     model.train()

#     for batch_dialogues, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
#         batch_labels = batch_labels.to(device)  # Move the labels to the device
#         logits, e = model(batch_dialogues)
#         loss = loss_function(logits.view(-1, num_classes), batch_labels.view(-1))

#         # Normalize the loss
#         loss = loss / accumulation_steps
#         loss.backward()

#         accumulation_step += 1
#         if accumulation_step % accumulation_steps == 0:
#             # Update the model parameters only when the accumulation step is reached
#             optimizer.step()
#             optimizer.zero_grad()
#             accumulation_step = 0

#         epoch_loss += loss.item() * accumulation_steps

#     print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / len(train_loader)}")

#     # Evaluation loop
#     model.eval()

#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch_dialogues, batch_labels in tqdm(test_loader, desc="Testing"):
#             logits, e = model(batch_dialogues)
#             preds = torch.argmax(logits, dim=2).cpu().numpy()
#             labels = batch_labels.cpu().numpy()
#             all_preds.extend(preds)
#             all_labels.extend(labels)

#     all_labels = np.concatenate(all_labels, axis=None)
#     all_preds = np.concatenate(all_preds, axis=None)

#     # Ignore the padded labels
#     mask = [label != -1 for label in all_labels]
#     all_preds = [all_preds[i] for i, mask_value in enumerate(mask) if mask_value]
#     all_labels = [all_labels[i] for i, mask_value in enumerate(mask) if mask_value]

#     print("\nTest set classification report:")
#     print(classification_report(all_labels, all_preds, zero_division=0))

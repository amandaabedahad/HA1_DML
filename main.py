from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from classes import SentimentClassifier, SwedishSentiDataset, train_epoch, eval_model
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import defaultdict


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = SwedishSentiDataset(
        text_reviews=df.text.to_numpy(),
        true_labels=df.sentiment.factorize()[0],
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0
    )

torch.cuda.empty_cache()
# CONSTANTS
BATCH_SIZE = 16
EPOCHS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")
dropout_rate_class_layer = 0.3
n_classes = 2  # positive or negative

tok = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
model_name = 'KB/bert-base-swedish-cased'

train_data = pd.read_csv('swedish_sentiment/train.csv')
train_data = train_data[:500]

test_data = pd.read_csv('swedish_sentiment/test.csv')[:500]
val_data = pd.read_csv('swedish_sentiment/dev.csv')[:500]

seq_len = [len(i.split()) for i in train_data["text"]]
sns.histplot(seq_len)
plt.title("Histogram of length of sentences")
plt.xlabel("length sentence")
plt.savefig("plots/hist_length_sentences.png")
max_length_sentence = 100

train_data_loader = create_data_loader(train_data, tok, max_length_sentence, BATCH_SIZE)
val_data_loader = create_data_loader(val_data, tok, max_length_sentence, BATCH_SIZE)
test_data_loader = create_data_loader(test_data, tok, max_length_sentence, BATCH_SIZE)

# NETWORK TO USE WITH TRANSFER LEARNING
net_model = SentimentClassifier(n_classes=n_classes,
                                val_dropout=dropout_rate_class_layer,
                                PRE_TRAINED_MODEL_NAME=model_name).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = AdamW(net_model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

# TRAAAAIn

history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(
        net_model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        train_data.shape[0]
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        net_model,
        val_data_loader,
        loss_fn,
        device,
        val_data.shape[0]
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:
        torch.save(net_model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);
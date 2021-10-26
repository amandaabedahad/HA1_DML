from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from utils_bert_pretrained import SentimentClassifier, SwedishSentiDataset, train_epoch, eval_model, create_data_loader
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import defaultdict
import numpy as np


torch.cuda.empty_cache()
# CONSTANTS
BATCH_SIZE = 32
EPOCHS = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")
dropout_rate_class_layer = 0.3
n_classes = 2  # positive or negative
max_length_sentence = 50

model_name = 'KB/bert-base-swedish-cased'
# model_name = 'bert-base-multilingual-cased'
tok = AutoTokenizer.from_pretrained(model_name)
pretrained_bert = AutoModel.from_pretrained(model_name)

for param in pretrained_bert.parameters():
    param.requires_grad = False

train_data = pd.read_csv('swedish_sentiment/train.csv')
train_data = train_data

test_data = pd.read_csv('swedish_sentiment/test.csv')
val_data = pd.read_csv('swedish_sentiment/dev.csv')

seq_len = np.array([len(i.split()) for i in train_data["text"]])
seq_len_val = np.array([len(i.split()) for i in val_data["text"]])
seq_len_test = np.array([len(i.split()) for i in test_data["text"]])

ind_train = np.argwhere(seq_len <= max_length_sentence)
ind_val = np.argwhere(seq_len_val <= max_length_sentence)
ind_test = np.argwhere(seq_len_test <= max_length_sentence)

train_data = train_data.iloc[ind_train.flatten()]
val_data = val_data.iloc[ind_val.flatten()]
test_data = test_data.iloc[ind_test.flatten()]
plt.rcParams.update({'font.size': 14})
sns.histplot(seq_len)
plt.title("Histogram of review length")
plt.xlabel("Number of tokens")
plt.savefig("plots/hist_length_sentences.png")


train_data_loader = create_data_loader(train_data, tok, max_length_sentence, BATCH_SIZE)
val_data_loader = create_data_loader(val_data, tok, max_length_sentence, BATCH_SIZE)
test_data_loader = create_data_loader(test_data, tok, max_length_sentence, BATCH_SIZE)

# NETWORK TO USE WITH TRANSFER LEARNING


net_model = SentimentClassifier(n_classes=n_classes,
                                val_dropout=dropout_rate_class_layer,
                                pretrained_bert=pretrained_bert).to(device)

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
    history['train_acc'].append(train_acc.cpu().detach().numpy())
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc.cpu().detach().numpy())
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:
        torch.save(net_model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc
        test_acc, _ = eval_model(
            net_model,
            test_data_loader,
            loss_fn,
            device,
            len(test_data)
        )
        print(f"test_acc:{test_acc.item()}")

plt.figure()
plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title(f'Training history, model: {model_name}')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])
plt.savefig('plots/acc_plot.png')

from transformers import AutoModel, AutoTokenizer
import pandas as pd
from classes import SentimentClassifier, SwedishSentiDataset
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = SwedishSentiDataset(
    text_reviews=df.text.to_numpy(),
    true_labels=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tok = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
model_name = 'KB/bert-base-swedish-cased'

train_data = pd.read_csv('swedish_sentiment/train.csv')
test_data = pd.read_csv('swedish_sentiment/test.csv')
val_data = pd.read_csv('swedish_sentiment/dev.csv')

seq_len = [len(i.split()) for i in train_data["text"]]
sns.histplot(seq_len)
plt.title("Histogram of length of sentences")
plt.xlabel("length sentence")
plt.savefig("plots/hist_length_sentences.png")
max_length_sentence = 200


n_classes = 2  # positive or negative
# NETWORK TO USE WITH TRANSFER LEARNING
net = SentimentClassifier(n_classes=n_classes,
                          val_dropout=0.3,
                          PRE_TRAINED_MODEL_NAME=model_name).to(device)


batch_size = 32


train_data_loader = create_data_loader(train_data, tok, max_length_sentence, batch_size)
val_data_loader = create_data_loader(val_data, tok, max_length_sentence, batch_size)
test_data_loader = create_data_loader(test_data, tok, max_length_sentence, batch_size)


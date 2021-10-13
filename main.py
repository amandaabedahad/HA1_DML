from transformers import AutoModel, AutoTokenizer
import pandas as pd

tok = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')

train_data = pd.read_csv('swedish_sentiment/train.csv')
test_data = pd.read_csv('swedish_sentiment/test.csv')
val_data = pd.read_csv('swedish_sentiment/dev.csv')
print(train_data.info)


def train_epoch():
    pass
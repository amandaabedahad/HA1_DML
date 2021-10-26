from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as torch_nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.data import get_tokenizer
from torch.utils.data import dataset, Dataset
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

tokenizer = get_tokenizer(tokenizer=None, language='se')
train_data = pd.read_csv('swedish_sentiment/train.csv')
test_data = pd.read_csv('swedish_sentiment/test.csv')
val_data = pd.read_csv('swedish_sentiment/dev.csv')

vocab = build_vocab_from_iterator(map(tokenizer, train_data['text']), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])


def data_process(data: dataset.IterableDataset) -> Tensor:
    data['text'] = data['text'].apply(lambda x: torch.tensor(vocab(tokenizer(x))))
    data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    return data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length_sentence = 50
seq_len = np.array([len(i.split()) for i in train_data["text"]])
seq_len_val = np.array([len(i.split()) for i in val_data["text"]])
seq_len_test = np.array([len(i.split()) for i in test_data["text"]])

ind_train = np.argwhere(seq_len <= max_length_sentence)
ind_val = np.argwhere(seq_len_val <= max_length_sentence)
ind_test = np.argwhere(seq_len_test <= max_length_sentence)

train_data = train_data.iloc[ind_train.flatten()]
val_data = val_data.iloc[ind_val.flatten()]
test_data = test_data.iloc[ind_test.flatten()]

train_data = data_process(train_data)
val_data = data_process(val_data)
test_data = data_process(test_data)


class Our_dataset(Dataset):
    def __init__(self, dataset_df):
        self.dataset_df = dataset_df

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tokenized_data = self.dataset_df['text'].iloc[idx]
        target = self.dataset_df['sentiment'].iloc[idx]
        sample = {'text': tokenized_data, 'sentiment': target}
        return sample


dataset_train = Our_dataset(train_data)
dataset_val = Our_dataset(val_data)
dataset_test = Our_dataset(test_data)


def my_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item['text'] for item in batch]
    padded_data = pad_sequence(data).T
    targets = [item['sentiment'] for item in batch]
    return [padded_data, targets]
batch_size = 32
train_loader = DataLoader(dataset_train,
                      batch_size=batch_size,
                      shuffle=True,
                      collate_fn=my_collate, # use custom collate function here
                      pin_memory=True)

val_loader = DataLoader(dataset_test,
                      batch_size=batch_size,
                      shuffle=True,
                      collate_fn=my_collate, # use custom collate function here
                      pin_memory=True)

test_loader = DataLoader(dataset_test,
                      batch_size=batch_size,
                      shuffle=True,
                      collate_fn=my_collate, # use custom collate function here
                      pin_memory=True)


class Transformer(torch_nn.Module):
    def __init__(self, dim, heads, depth, seq_length, num_tokens, num_classes, device):
        super().__init__()
        self.device = device

        self.num_tokens = num_tokens

        self.pos_emb = torch_nn.Embedding(seq_length, dim)
        self.token_emb = torch_nn.Embedding(num_tokens, dim)

        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=depth)

        self.output_map = torch_nn.Linear(dim, num_classes)

    def forward(self, x):
        """Transformer forward method

        Args:
            x Tensor(batch_size, seq_length): Word indices representing sequence of words.
        Returns:
            Tensor(batch_size, num_classes): Log logits
        """
        tokens = self.token_emb(x)
        batch_size, seq_length, dim = tokens.size()

        # Note that we create a completely new tensor which must be moved to the proper device.
        # This is why we must store the device in self.device.
        pos = torch.arange(seq_length, device=self.device)
        pos = self.pos_emb(pos)[None, :, :].expand(batch_size, seq_length, dim)

        x = tokens + pos
        x = self.transformer_encoder(x)

        x = self.output_map(x.mean(dim=1))
        return F.log_softmax(x, dim=1)


from time import time


def train_epoch(model, train_loader, optimizer, scheduler, max_seq_len):
    """Train epoch"""
    train_loss = AccumulatingMetric()
    train_acc = AccumulatingMetric()
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_, label = batch[0].to(device), torch.tensor(batch[1]).to(device)

        #input_ = _truncate_input(input_, max_seq_len)
        pred = model(input_)
        loss = F.nll_loss(pred, label)
        loss.backward()
        train_loss.add(loss.item())

        train_acc.add(accuracy(pred, label))

        # Gradient clipping is a way to ensure
        # torch_nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return train_loss.avg(), train_acc.avg()


def validate_epoch(model, val_loader, max_seq_len):
    val_loss = AccumulatingMetric()
    val_acc = AccumulatingMetric()
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_, label = batch[0].to(device), torch.tensor(batch[1]).to(device)

            #input_ = _truncate_input(input_, max_seq_len)
            pred = model(input_)
            val_loss.add(F.nll_loss(pred, label).item())

            val_acc.add(accuracy(pred, label))

    return val_loss.avg(), val_acc.avg()  # TODO: loss


def accuracy(pred, label):
    hard_pred = pred.argmax(1)
    return (hard_pred == label).float().mean().item()


def _truncate_input(input_, max_seq_len):
    if input_.size(1) > max_seq_len:
        input_ = input_[:, :max_seq_len]
    return input_


class AccumulatingMetric:
    """Accumulate samples of a metric and automatically keep track of the number of samples."""

    def __init__(self):
        self.metric = 0.0
        self.counter = 0

    def add(self, value):
        self.metric += value
        self.counter += 1

    def avg(self):
        return self.metric / self.counter




num_tokens = len(vocab)
max_length = 50
embedding_size = 512
num_heads = 8
num_classes = 2
depth = 6
history = defaultdict(list)
model = Transformer(
    dim=embedding_size,
    heads=num_heads,
    depth=depth,
    seq_length=max_length,
    num_tokens=num_tokens,
    num_classes=num_classes,
    device=device)

model.to(device)

lr = 1e-4
lr_warmup = 1e4
num_epochs = 10

#train_loader, test_loader = get_loaders(num_tokens, batch_size, device)

optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
# A scheduler is a principled way of controlling (often decreasing) the learning rate as time progresses.
# Read more: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda i: min(i / (lr_warmup / batch_size), 1.0)
)

print("Starting training")
best_val_acc = 0
for epoch in range(1, num_epochs + 1):
    start = time()
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, max_length)
    val_loss, val_acc = validate_epoch(model, val_loader, max_length)
    end = time()
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'best_model_state_our_transformer.bin')
        best_val_acc = val_acc
    print(
        "Epoch: {}/{}: time: {:.1f}, train loss: {:.3f}, train acc: {:.3f}, val. loss {:.3f}, val. acc: {:.3f}".format(
            epoch, num_epochs, end - start, train_loss, train_acc, val_loss, val_acc
        )
    )
plt.figure()
plt.plot(history['train_loss'], label='train loss')
plt.plot(history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss transformer')
plt.legend()
plt.savefig('plots/loss_transformer.png')
plt.figure()
plt.plot(history['train_acc'], label='train acc')
plt.plot(history['val_acc'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy transformer')
plt.legend()
plt.savefig('plots/acc_transformer.png')
print("You have now trained a transformer!")
test_acc = AccumulatingMetric()
model.eval()

with torch.no_grad():
    for batch in test_loader:
        input_, label = batch[0].to(device), torch.tensor(batch[1]).to(device)

        #input_ = _truncate_input(input_, max_seq_len)
        pred = model(input_)

        test_acc.add(accuracy(pred, label))

avg_test_acc = test_acc.avg()
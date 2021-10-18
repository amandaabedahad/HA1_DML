import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.utils.data import Dataset, DataLoader





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


class SwedishSentiDataset(Dataset):
    def __init__(self, text_reviews, true_labels, tokenizer, max_len):
        self.text_reviews = text_reviews
        self.true_labels = true_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text_reviews)

    def __getitem__(self, item):
        review = str(self.text_reviews[item])
        target = self.true_labels[item]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'true_labels': torch.tensor(target, dtype=torch.long)
        }
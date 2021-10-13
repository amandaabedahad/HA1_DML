import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, val_dropout, PRE_TRAINED_MODEL_NAME):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=val_dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


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


class Train():
    pass
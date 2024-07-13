import torch.nn as nn
from torch.utils.data import Dataset

class IslamicDataset(Dataset):
    def __init__(self, texts, sources, tokenizer, max_length=512):
        self.texts = [chunk for text_chunks in texts for chunk in text_chunks]
        self.sources = [source for source, text_chunks in zip(sources, texts) for _ in text_chunks]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        source = self.sources[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text,
            'source': source
        }

class IslamicQAModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.3)
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)
        self.retrieval = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        retrieval_score = self.retrieval(sequence_output[:, 0, :]).squeeze(-1)
        return start_logits, end_logits, retrieval_score
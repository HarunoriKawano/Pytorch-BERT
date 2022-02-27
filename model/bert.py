import torch
from torch import nn

from model.transformer import TransformerLayer
from model.embedding import BERTEmbedding


class BertModel(nn.Module):
    """Bidirectional Encoder Representations from Transformers

    embedding -> encoder -> pool(cls processing)

    Attributes:
        embedding (BERTEmbedding): Perform vector embedding for word, position, and token type.
        encoder (BertEncoder): Extracting features of a sentence
        pool (BertPool): Process the cls tokens separately to make them easier to handle.
    """
    def __init__(self, vocab_size=30522, max_seq_len=512, type_vocab_size=2,
                 feature_size=768, attention_head_num=12, bert_layer_num=12, hidden_size=3072,
                 device=torch.device('cpu')):
        super(BertModel, self).__init__()
        self.embedding = BERTEmbedding(vocab_size, max_seq_len, type_vocab_size, feature_size, device=device)
        self.encoder = BertEncoder(feature_size, hidden_size, attention_head_num, bert_layer_num)
        self.pool = BertPool(feature_size)

    def forward(self, inputs, token_type_ids=None, mask=None):
        """
        Args:
            inputs (torch.LongTensor(batch_size, max_seq_len)): word ids in sentence
            token_type_ids (torch.LongTensor(batch_size, max_seq_len) or None): token types of the word
            mask (torch.LongTensor(batch_size, max_seq_len) or None): mask of padding word
        Returns:
            encoded_layers (torch.FloatTensor(batch_size, max_seq_len, feature_size)): output of BERT encoder
            pooled_first_token (torch.FloatTensor(batch_size, feature_size)): feature of cls
        """
        if mask is None:
            mask = torch.ones_like(inputs)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(inputs)

        extended_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_mask = extended_mask.to(dtype=torch.float32)
        # Set the mask part to -10000 so that it becomes a mask when the softmax is calculated.
        extended_mask = (1.0 - extended_mask) * -10000.0

        embedding_out = self.embedding(inputs, token_type_ids)
        encoded_layers = self.encoder(embedding_out, extended_mask)

        pooled_first_token = self.pool(encoded_layers)

        return encoded_layers, pooled_first_token


class BertEncoder(nn.Module):
    """Encoder part in BERT

    standard: TransformerLayer ×　12

    Attributes:
        bert_layers (nn.ModuleList(TransformerLayer × bert_layer_num)): The transformer layers that makes up BERT
    """
    def __init__(self, feature_size, hidden_size, attention_head_num, bert_layer_num=12):
        super(BertEncoder, self).__init__()
        self.bert_layers = nn.ModuleList([TransformerLayer(feature_size, hidden_size, attention_head_num) for _ in range(bert_layer_num)])

    def forward(self, inputs, mask=None):
        """
        Args:
            inputs (torch.FloatTensor(batch_size, max_seq_len, feature_size))
            mask (torch.LongTensor(batch_size, max_seq_len) or None)

        Returns:
            torch.FloatTensor(batch_size, max_seq_len, feature_size)
        """
        out = inputs
        for layer in self.bert_layers:
            out = layer(out, mask)

        return out


class BertPool(nn.Module):
    """Processing of cls tokens

    Perform feature extraction on cls tokens.

    Attributes:
        linear (nn.Linear)
    """
    def __init__(self, feature_size):
        super(BertPool, self).__init__()
        self.linear = nn.Linear(feature_size, feature_size)

    def forward(self, inputs):
        """
            Args:
                inputs (torch.FloatTensor(batch_size, max_seq_len, feature_size))

            Returns:
                torch.FloatTensor(batch_size, feature_size)
        """
        first_token = inputs[:, 0, :]
        out = torch.tanh(self.linear(first_token))

        return out

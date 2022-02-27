from torch import nn
import torch.nn.functional as F

from model.attention import AttentionLayer


class TransformerLayer(nn.Module):
    """transformer layer in bert layer

    attention layer -> feed forward -> layer normalization

    Attributes:
        attention_layer (AttentionLayer): attention layer in transformer layer
        feed_forward (TransformerFeedForward): feed forward in transformer layer
        layer_norm (nn.LayerNorm): layer normalization
    """
    def __init__(self, feature_size, hidden_size, attention_head_num):
        super(TransformerLayer, self).__init__()

        self.attention_layer = AttentionLayer(feature_size, attention_head_num)
        self.feed_forward = TransformerFeedForward(feature_size, hidden_size)
        self.layer_norm = nn.LayerNorm(feature_size)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (torch.FloatTensor(batch_size, max_seq_len, feature_size))
            mask (torch.LongTensor(batch_size, max_seq_len) or None)

        Returns:
            torch.FloatTensor(batch_size, max_seq_len, feature_size)
        """
        attention = self.attention_layer(inputs, mask)
        out = self.feed_forward(attention)
        out = self.layer_norm(out + inputs)

        return out


class TransformerFeedForward(nn.Module):
    """feed forward layer in Transformer

    linear -> linear -> dropout

    Attributes:
        linear1 (nn.Linear)
        linear2 (nn.Linear)
        dropout (nn.Dropout)
    """
    def __init__(self, feature_size, hidden_size, dropout_rate=0.1):
        super(TransformerFeedForward, self).__init__()

        self.linear1 = nn.Linear(feature_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, feature_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.FloatTensor(batch_size, max_seq_len, feature_size))

        Returns:
            torch.FloatTensor(batch_size, max_seq_len, feature_size)
        """
        out = F.gelu(self.linear1(inputs))
        out = self.dropout(self.linear2(out))

        return out

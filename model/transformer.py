from torch import nn
import torch.nn.functional as F

from model.attention import AttentionLayer


class TransformerLayer(nn.Module):
    def __init__(self, feature_size, hidden_size, attention_head_num):
        super(TransformerLayer, self).__init__()

        self.attention_layer = AttentionLayer(feature_size, attention_head_num)
        self.feed_forward = TransformerFeedForward(feature_size, hidden_size)
        self.layer_norm = nn.LayerNorm(feature_size)

    def forward(self, inputs, mask=None):
        attention = self.attention_layer(inputs, mask)
        out = self.feed_forward(attention)
        out = self.layer_norm(out + inputs)

        return out


class TransformerFeedForward(nn.Module):
    def __init__(self, feature_size=512, hidden_size=3072, dropout_rate=0.1):
        super(TransformerFeedForward, self).__init__()

        self.linear1 = nn.Linear(feature_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, feature_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        out = F.gelu(self.linear1(inputs))
        out = self.dropout(self.linear2(out))

        return out

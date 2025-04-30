import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(self.max_len, d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # 形状 (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # seq_len = x.size(0)
        # if seq_len > self.max_len:
        #     pe = self.generate_positional_encodings(seq_len)
        # else:
        #     pe = self.pe[:seq_len]
        # x = x + pe
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1, max_seq_length=5000):
        """
        定义基于 Transformer Encoder 的语言模型
        参数说明：
        - vocab_size: 词汇表大小
        - d_model: 词嵌入及 Transformer 模型维度
        - nhead: 多头注意力的头数
        - num_layers: Transformer Encoder 层数
        - dim_feedforward: 前馈网络隐藏层维度
        - dropout: dropout 概率
        - max_seq_length: 最大序列长度（用于位置编码）
        """
        super(TransformerLanguageModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask):
        """
        src: 输入序列，形状 (seq_len, batch_size)
        src_mask: 掩码矩阵，形状 (seq_len, seq_len)
        """
        # 嵌入并缩放
        emb = self.embedding(src) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)  # 使用位置编码
        output = self.transformer_encoder(emb, src_mask)  # 使用掩码
        logits = self.fc_out(output)
        return logits

def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
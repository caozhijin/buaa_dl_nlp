import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from transformer import *
from dataPrepare import *

# 选择设备（GPU 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, full_dataset):
    epochs = 500
    batch_size = 512
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 划分数据集为训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    best_loss = float('inf')
    patience = 3
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_dataloader):
            # data, targets 的形状为 (batch_size, seq_length)
            # 转置为 (seq_length, batch_size) 符合 Transformer 的输入要求
            data = data.transpose(0, 1).to(device)
            targets = targets.transpose(0, 1).to(device)

            optimizer.zero_grad()
            seq_len = data.size(0)
            # 生成当前序列长度对应的掩码
            src_mask = generate_square_subsequent_mask(seq_len).to(device)
            output = model(data, src_mask)  # 输出形状为 (seq_length, batch_size, vocab_size)
            loss = criterion(output.view(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 200 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # 在验证集上评估模型
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data, targets in val_dataloader:
                # data, targets 的形状为 (batch_size, seq_length)
                # 转置为 (seq_length, batch_size) 符合 Transformer 的输入要求
                data = data.transpose(0, 1).to(device)
                targets = targets.transpose(0, 1).to(device)

                seq_len = data.size(0)
                src_mask = generate_square_subsequent_mask(seq_len).to(device)
                output = model(data, src_mask)
                loss = criterion(output.view(-1, vocab_size), targets.reshape(-1))
                val_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch} 完成，训练集平均 Loss: {avg_loss:.4f}, 验证集平均 Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), 'transformer.keras')
        else:
            wait += 1
            lr = lr * 0.5
            if wait >= patience:
                print("Early stopping due to no improvement")
                break


def generate_text(model, prompt, vocab, seq_length=50, gen_length=100):
    """
    根据提示词生成文本
    参数：
    - model: 已训练好的 Transformer 语言模型
    - prompt: 用户输入的提示词（字符串）
    - vocab: 词汇表（词->索引字典）
    - seq_length: 训练时使用的序列长度（用于生成掩码）
    - gen_length: 生成的新词数量
    """
    model.eval()
    # 构造索引到词的反向映射
    idx2word = {idx: word for word, idx in vocab.items()}

    # 对提示词进行分词和数值化
    prompt_tokens = list(jieba.cut(prompt))
    prompt_ids = [vocab.get(word, vocab["<unk>"]) for word in prompt_tokens]

    # 将提示词转换为 tensor，形状 (seq_len, 1)
    input_seq = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(1).to(device)
    generated = prompt_tokens.copy()

    with torch.no_grad():
        for _ in range(gen_length):
            seq_len = input_seq.size(0)
            src_mask = generate_square_subsequent_mask(seq_len).to(device)
            output = model(input_seq, src_mask)
            # 取最后一个时间步的输出，预测下一个词
            last_logits = output[-1, 0, :]
            probs = F.softmax(last_logits, dim=0)
            next_token_id = torch.multinomial(probs, 1).item()
            next_word = idx2word.get(next_token_id, "<unk>")
            generated.append(next_word)
            # 将生成的词添加到输入序列中
            next_token = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            input_seq = torch.cat([input_seq, next_token], dim=0)
            # 如果生成了句子结束符，则终止生成
            if next_word == "<eos>":
                break
    return ' '.join(generated)


if __name__ == '__main__':
    file = data_path + '笑傲江湖.txt'
    d = get_file_data(file)
    vocab = build_vocab( d, min_freq=1)
    token_ids = numericalize( d, vocab)
    seq_length = 50
    dataset = LanguageModelDataset(token_ids, seq_length)

    vocab_size = len(vocab)  # 词汇表大小
    d_model = 512  # 模型维度
    nhead = 8  # 多头注意力的头数
    num_layers = 4  # Transformer 层数
    dim_feedforward = 2048  # 前馈网络隐藏层维度
    dropout = 0.1  # dropout 概率
    max_seq_length = 512  # 最大序列长度

    model = TransformerLanguageModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, max_seq_length)

    train(model, dataset)
    model.load_state_dict(torch.load('transformer.keras'))
    model.to(device)

    # 生成文本
    initial_text = '青衣剑士连劈三剑，锦衫剑士一一格开。青衣剑士一声吒喝，长剑从左上角直划而下'
    generated_text = generate_text(model, initial_text, vocab, seq_length=seq_length, gen_length=100)
    print(generated_text)


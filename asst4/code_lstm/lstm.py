import jieba
import tensorflow as tf
from tensorflow.keras import optimizers, layers, models, callbacks
from dataPrepare import *

# 检查 GPU 是否可用
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 设置可见的 GPU 设备为 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 限制 TensorFlow 只使用设备 1
        tf.config.set_visible_devices([gpus[1]], 'GPU')
        print("Visible GPUs: ", [gpu.name for gpu in tf.config.get_visible_devices('GPU')])
    except RuntimeError as e:
        print(e)

# 设置 GPU 内存增长
visible_gpus = tf.config.get_visible_devices('GPU')
if visible_gpus:
    try:
        for gpu in visible_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# 回调函数列表
callbacks_list = [
    callbacks.ModelCheckpoint(
        filepath='text_gen.keras',
        monitor='loss',
        save_best_only=True,
    ),
    callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=1,
    ),
    callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
    ),
]

# 样本生成函数
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def train(x, y, tokens, epochs=300):
    x = np.asarray(x)
    y = np.asarray(y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # 分割数据集为训练集和测试集
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    with tf.device('/GPU:1'):  # 指定使用第一个可见的 GPU
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(tokens), 256),
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.LSTM(256),
            tf.keras.layers.Dense(len(tokens), activation='softmax')
        ])

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # 训练模型
        model.fit(
            train_dataset,
            validation_data=test_dataset,  # 在测试集上评估模型
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1
        )

    return model

# 文本生成函数
def generate_text(model, tokens, tokens_indices, initial_text, max_length=100, temperature=1.0):
    text = initial_text
    text_cut = list(jieba.cut(text))[:60]
    print(text, end='')

    with tf.device('/GPU:0'):  # 指定使用第一个可见的 GPU
        for i in range(max_length):
            sampled = np.zeros((1, 60))
            for idx, token in enumerate(text_cut):
                if token in tokens_indices:
                    sampled[0, idx] = tokens_indices[token]

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature=temperature)
            next_token = tokens[next_index]
            print(next_token, end='')

            text_cut = text_cut[1:] + [next_token]

# 主函数
if __name__ == '__main__':
    file = data_path + '笑傲江湖.txt'
    d = get_file_data(file)
    _x, _y, _tokens, _tokens_indices = get_dataset(d)

    # 训练模型
    model = train(_x, _y, _tokens, _tokens_indices, epochs=300)

    # 保存模型为 .keras 格式
    model.save('lstm.keras')

    # 生成文本
    initial_text = '青衣剑士连劈三剑，锦衫剑士一一格开。青衣剑士一声吒喝，长剑从左上角直划而下，势劲力急。锦衫剑士身手矫捷，向后跃开，避过了这剑。他左足刚着地，身子跟着弹起，刷刷两剑，向对手攻去。青衣剑士凝里不动，嘴角边微微冷笑，长剑轻摆，挡开来剑。'
    generate_text(model, _tokens, _tokens_indices, initial_text, max_length=100, temperature=1.0)

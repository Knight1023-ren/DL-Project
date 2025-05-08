# 导入模块
from torch.autograd import Variable
from rnn import Word_embedding
import torch.optim as optim
from rnn import RNN_model
from tqdm import tqdm
import collections
import numpy as np
import torch

start_token = 'G'  # 序列生成的起始标记为'G'
end_token = 'E'  # 序列生成的结束标记为'E'
batch_size = 64  # 一次处理64个样本


def process_poems(file_name):
    """
    从文件中读取诗歌内容，并进行处理使其转换为何时的输入格式。
    :param file_name:输入文件路径。
    """
    poems = []  # 初始化存储唐诗的列表。
    # 打开文件并逐行读取内容。
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                parts = line.strip().split(':', 1)
                if len(parts) != 2:
                    continue
                title, content = parts  # 将每行按':'分割为标题和内容。
                content = content.replace(' ', '')  # 移除空格。
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content: continue  # 跳过特殊符号和起始、终止符号
                if len(content) < 5 or len(content) > 80: continue  # 跳过长度小于5或大于80的诗句。
                content = start_token + content + end_token  # 在内容前后添加起始和终止符号。
                poems.append(content)  # 添加到poems列表中
            except ValueError as e:
                print(e)
    poems = sorted(poems, key=lambda line: len(line))  # 根据每首诗的字数对poems列表进行排序
    # 统计每个字出现次数
    all_words = []  # 初始化存储所有诗的所有字的空列表。
    for poem in poems:
        all_words += [word for word in poem]  # 收集所有诗歌中的单词
    counter = collections.Counter(all_words)  # 统计词和词频
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 按词频降序排序
    words, _ = zip(*count_pairs)  # 将单词列表和词频分开
    words = words[:len(words)] + (' ',)  # 在单词列表末尾添加空格
    word_int_map = dict(zip(words, range(len(words))))  # 创建字典，将每个单词映射到一个唯一的整数
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]  # 使用word_int_map将每首诗的内容转换为整数序列
    # 返回整数序列的列表（每个序列代表一首诗，poems_vector）、单词到整数的映射字典（word_int_map）、所有出现的单词（words）
    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    """
    用于从给定的诗向量列表中生成训练批次，这些批次可以用于训练神经网络模型。
    :param batch_size:每个批次中包含的样本数量。
    :param poems_vec:诗歌的向量表示列表，每个向量一首诗。
    """
    n_chunk = len(poems_vec) // batch_size  # 计算批次数量
    x_batches = []  # 初始化存储输入数据批次。
    y_batches = []  # 初始化存储目标数据批次。
    # 从给定的诗向量列表中生成训练批次
    for i in range(n_chunk):
        start_index = i * batch_size  # 当前批次的起始索引
        end_index = start_index + batch_size  # 当前批次的结束索引
        x_data = poems_vec[start_index:end_index]  # 提取当前批次的数据
        y_data = []  # 初始化存储当前批次的目标数据的空列表
        # 遍历当前批次的每个样本
        for row in x_data:
            y = row[1:]  # 提取当前样本的第一个元素之后的所有元素
            y.append(row[-1])  # 将当前样本的最后一个元素再次添加到y中（确保目标序列的长度与输入序列的长度一致）
            y_data.append(y)  # 将生成的y添加到y_data中
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)  # 添加当前批次的输入数据
        y_batches.append(y_data)  # 添加当前批次的目标数据
    return x_batches, y_batches


# 训练一个LSTM模型生成诗歌。
def run_training():
    poems_vector, word_to_int, vocabularies = process_poems('poems.txt')  # 数据预处理，从poems.txt文件中加载和处理数据。
    print("数据加载完成。")
    BATCH_SIZE = 100  # 设置批次大小为100
    torch.manual_seed(5)  # 设置随机种子确保实验可重复性。
    # 初始化Word_embedding对象，用于将单词转换为向量。
    word_embedding = Word_embedding(vocab_length=len(word_to_int) + 1, embedding_dim=100)
    rnn_model = RNN_model(batch_sz=BATCH_SIZE, vocab_len=len(word_to_int) + 1, word_embedding=word_embedding,
                          embedding_dim=100, lstm_hidden_dim=128)  # 初始化RNN_model对象，用于生成诗歌。
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)  # 初始化Adam优化器，用于更新模型参数。
    loss_fun = torch.nn.NLLLoss()  # 使用负对数似然损失函数NLLLoss。
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控损失，越小越好
        factor=0.5,  # 学习率降低因子
        patience=2,  # 等待2个epoch无改善再降低学习率
        min_lr=1e-5  # 最小学习率
    )
    rnn_model.load_state_dict(torch.load('poem_generator_rnn.pt'))  # 加载预训练权重（如果有的话）
    min_loss = float('inf')
    for epoch in tqdm(range(30)):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        epoch_loss = 0
        # 训练所有批次
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch]
            batch_loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype=np.int64)
                y = np.array(batch_y[index], dtype=np.int64)
                x = Variable(torch.from_numpy(np.expand_dims(x, axis=1)))
                y = Variable(torch.from_numpy(y))
                pre = rnn_model(x)
                batch_loss += loss_fun(pre, y)
            batch_loss = batch_loss / BATCH_SIZE
            epoch_loss += batch_loss.item()  # 计算当前批次的平均损失
            # 反向传播和参数更新
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1)
            optimizer.step()
        avg_epoch_loss = epoch_loss / n_chunk  # 计算本轮的平均损失
        scheduler.step(avg_epoch_loss)  # 使用训练损失更新学习率
        print(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}")  # 输出每一轮的损失
        # 保存损失最小的模型
        if avg_epoch_loss < min_loss:
            min_loss = avg_epoch_loss
            torch.save(rnn_model.state_dict(), 'poem_generator_rnn.pt')
            print(f"已保存最佳模型。")


def to_word(predict, vocabs):
    """
    用于将模型的预测结果转换为对应的汉字。
    :param predict:模型的预测结果。
    :param vocabs:词汇表，每个元素对应一个单词或汉字。
    """
    sample = np.argmax(predict)  # 获取最大值的索引
    if sample >= len(vocabs): sample = len(vocabs) - 1  # 如果索引超出范围则取最后一个有效索引
    return vocabs[sample]  # 从vocab中获取对应的单词或汉字。


def pretty_print_poem(poem):
    """
    格式化并打印诗，使其输出工整。
    :param poem:一首诗。
    """
    shige = []  # 初始化列表，用于存储处理后的诗歌。
    # 过滤起始和结束标记。
    for w in poem:
        if w == start_token or w == end_token: break
        shige.append(w)
    poem_sentences = poem.split('。')  # 将poem按句号进行分割，存储在列表中。
    for s in poem_sentences:
        if s != '' and len(s) > 10: print(s + '。')  # 输出符合句子长度大于10的诗句。


def gen_poem(begin_word):
    """
    用于生成一首诗。
    :param begin_word:生成诗歌的起始词。
    """
    poems_vector, word_int_map, vocabularies = process_poems('poems.txt')  # 加载并处理poem.txt中的诗歌数据
    word_embedding = Word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)  # 将单词转换为向量表示
    rnn_model = RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,
                          embedding_dim=100, lstm_hidden_dim=128)  # 初始化RNN_model对象，用于生成诗歌
    rnn_model.load_state_dict(torch.load('poem_generator_rnn.pt'))  # 从文件中加载模型权重
    # 指定开始的字
    poem = begin_word
    word = begin_word
    # 知道生成的word等于终止标记或诗歌长度超过30。
    while word != end_token:
        input = np.array([word_int_map[w] for w in poem], dtype=np.int64)  # 将当前poem转换为整数表示的数组
        input = Variable(torch.from_numpy(input))  # 转换为pytorch张量
        output = rnn_model(input, is_test=True)  # 在测试模式下使用rnn_model对输入进行预测
        word = to_word(output.data.tolist()[-1], vocabularies)  # 从模型的输出中获取最后一个预测结果，并将其转换为单词
        if word == 'E': continue
        poem += word  # 将新生成的word添加到poem中
        if len(poem) > 30: break
    print(poem)
    return poem


# run_training()  # 进行模型训练。如果不是训练阶段，请注释。

pretty_print_poem(gen_poem("日"))
pretty_print_poem(gen_poem("红"))
pretty_print_poem(gen_poem("山"))
pretty_print_poem(gen_poem("夜"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("海"))
pretty_print_poem(gen_poem("月"))
# pretty_print_poem(gen_poem("君"))

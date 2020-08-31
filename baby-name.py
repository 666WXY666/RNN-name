"""
@Copyright: Copyright (c) 2020 苇名一心 All Rights Reserved.
@Project: rnn-name
@Description: 用RNN网络为baby起名字
@Version: 1.0
@Author: 苇名一心
@Date: 2020-06-05 17:13
@LastEditors: 苇名一心
@LastEditTime: 2020-06-05 17:13
"""
from __future__ import unicode_literals, print_function, division

import glob
import math
import os
import random
import string
import time
from io import open

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import unicodedata

#################################################
# 全局参数
#################################################
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1
criterion = nn.NLLLoss()
learning_rate = 0.0005  # 学习率
max_length = 20
path = "data/names/*.txt"


#################################################
# 数据集操作
#################################################
# 将Unicode转为ASCII
# https://stackoverflow.com/a/518232/2809427
def Unicode_to_ASCII(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# 将行分割成数组, 并把 Unicode 转换成 ASCII 编码, 最后放进一个字典里 {category: [names ...]}
category_lines = {}
all_categories = []

for file_name in glob.glob(path):
    category = os.path.splitext(os.path.basename(file_name))[0]
    all_categories.append(category)
    line_list = open(file_name, encoding='utf-8').read().strip().split('\n')
    lines = [Unicode_to_ASCII(line) for line in line_list]
    category_lines[category] = lines

category_num = len(all_categories)

if category_num == 0:
    raise RuntimeError("未找到数据集！")

print("数据集类别：", category_num, all_categories)


#################################################
# 构建RNN网络
#################################################
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(category_num + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(category_num + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


#################################################
# 训练前的准备
#################################################
# 输入串从第一个字母到最后一个字母（不包括 EOS ）的 one-hot 矩阵
def generate_input_tensor(letters):
    tensor = torch.zeros(len(letters), 1, n_letters)
    for i in range(len(letters)):
        letter = letters[i]
        tensor[i][0][all_letters.find(letter)] = 1
    return tensor


# 目标的第k个字母到结尾（EOS）的 LongTensor
def generate_target_Tensor(letters):
    letter_indexes = [all_letters.find(letters[i]) for i in range(1, len(letters))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


# 类别的One-hot向量
def generate_category_tensor(category):
    i = all_categories.index(category)
    tensor = torch.zeros(1, category_num)
    tensor[0][i] = 1
    return tensor


# 利用辅助函数从数据集中获取随机的category和line
def random_pair():
    category = all_categories[random.randint(0, len(all_categories) - 1)]
    line = category_lines[category][random.randint(0, len(category_lines[category]) - 1)]
    return category, line


# 从随机的（category, line）对中生成 category, input, 和 target Tensor
def randomTrainingExample():
    category, line = random_pair()
    category_tensor = generate_category_tensor(category)
    input_tensor = generate_input_tensor(line)
    target_tensor = generate_target_Tensor(line)
    return category_tensor, input_tensor, target_tensor


#################################################
# 训练RNN网络
#################################################
def train(category_tensor, input_tensor, target_tensor):
    target_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_tensor[i], hidden)
        l = criterion(output, target_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_tensor.size(0)


# 秒转时间戳
def time_format(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# 从一个类中获取一个以start_letters开头的名字
def sample(category, start_letters='Al'):
    with torch.no_grad():
        category_tensor = generate_category_tensor(category)
        input = generate_input_tensor(start_letters)
        hidden = rnn.init_hidden()

        output_name = start_letters

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi_temp = output.topk(5)
            topi = topi_temp[0][0].item()
            if topi == n_letters - 1:
                break
            else:
                print("[", end="")
                for j in range(5):
                    if j != 4:
                        if topi_temp[0][j].item() >= n_letters - 1:
                            print(" ", end=",")
                        else:
                            print(all_letters[topi_temp[0][j].item()], end=",")
                    else:
                        if topi_temp[0][j].item() >= n_letters - 1:
                            print(" ", end="]\n")
                        else:
                            print(all_letters[topi_temp[0][j].item()], end="]\n")
                letter = all_letters[topi]
                output_name += letter
            input = generate_input_tensor(letter)
        print(output_name)


#################################################
# 开始运行
#################################################
rnn = RNN(n_letters, 128, n_letters)

all_losses = []
total_loss = 0

# 超参数参考值
# n_its = 100000
# print_every = 5000
# plot_every = 500

print("请分别输入总迭代次数,打印精度,绘图精度(按空格分隔): ", end="")
n_its, print_every, plot_every = map(int, input().split())
print("开始训练......")
start = time.time()
print("[时间戳]\t\t百分比\t\t已迭代数\t\t损失")
for it in range(1, n_its + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if it % print_every == 0:
        print("[%10s]  <%3d%%>  %10d  %.4f" % (time_format(start), it / n_its * 100, it, loss))

    if it % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0
print("训练完成！")

# 画出损失图像
print("绘制损失图像......")
plt.figure()
plt.plot(all_losses)
plt.show()

# 生成测试
print("请输入是否要进行网络采样测试(1,是;0,否): ", end="")
flag = int(input())
while flag:
    print("请输入类别和姓名首字母(按空格分割,例: female Al): ", end="")
    cat, letters = input().split()
    sample(cat, letters)
    print("要继续吗(1,继续;0,停止)？", end="")
    flag = int(input())

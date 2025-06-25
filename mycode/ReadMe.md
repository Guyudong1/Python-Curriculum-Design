1.DQN-training-model是训练文件，1000轮次下需要几分钟的训练时间，稍作等待。
2.DQN-training-model注意保留的位置，需要手动修改，以免报错。
3.init-Snake是贪吃蛇的游戏本体，方向键控制运行。
4.Final_Snake_DQN是最终的AI游玩文件，记得手动修改导入文件地址，以免报错。
5.DQN的input size和output size分别为151和4，158是由输入参数决定的，4是输出上下左右的控制决定的，非必要，不要修改，以免报错。
6.pth是用来保存agent模型的，建议不要直接和py文件混在一起，建议注意分层管理文件。
7.HumanvsAi，same是同屏竞争，DIF是分屏竞争

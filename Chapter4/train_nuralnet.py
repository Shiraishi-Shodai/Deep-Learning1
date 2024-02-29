import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from matplotlib import pyplot as plt
import japanize_matplotlib
from tqdm import tqdm

# データの読み込み
(X_train, t_train), (X_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = [] # ミニバッチ学習を行うごとに損失関数を計算し格納するリスト

# ハイパーパラメータの設定
iters_num = 100
train_size = X_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = [] # 損失関数
train_acc_list = [] # 学習時の正解率
test_acc_list = []  # テスト時の正解率

iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in tqdm(range(iters_num)):
    
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size) # (0以上60000未満のインデックスを取得)
    x_batch = X_train[batch_mask]  # batch_maskに対応する学習データを取得
    t_batch = t_train[batch_mask]  # batch_maskに対応する学習データラベルを取得
    
    
    # 勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch) # ドット積はかけられる数の列数とかける数の行数が等しければ計算可能なため、バッチ処理が可能
    # grad - network.gradient(x_batch, t_batch) # 高速版!
    
    
    # パラメータの更新
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]
        
    
    # 学習過程の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(X_train, t_train)
        test_acc = network.accuracy(X_test, t_test)
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print(f'train acc, test acc | {train_acc, test_acc}')
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01) -> None:
        # 重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
                            
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        
        self.params["b2"] = np.zeros(output_size)
        
    def predict(self, x):
        """各層での学習
        
        [Parameter]
        x : 入力データ
        
        [Return]
        Y : 出力層の値
        """
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        
        a1 = np.dot(x,W1 ) + b1
        z1 = sigmoid(a1)
        
        a2 = np.dot(z1, W2) + b2
        Y = softmax(a2)
        
        return Y
    
    def loss(self, x, t):
        """出力層の結果を元に損失関数を求める
        
        [Parameter]
        x : 入力データ
        t : 正解ラベル
        
        [Return]
        cross_entropy_error(y, t) : 損失関数の値
        """
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        """認識精度を確かめる
        
        [Parameter]
        x : 入力データ
        t : 正解ラベル
        
        [Return]
        """
        y = self.predict(x)
        # (バッチ処理をした場合は出力層の結果が行列になるから行ごとに計算)
        y = np.argmax(y, axis=1) # 取得したベクトルの最大値のインデックスを取得
        t = np.argmax(t, axis=1) # 答えのインデックスを取得
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        
        return accuracy
    
    def numerical_gradient(self, x, t):
        """それぞれの重みの勾配を計算
        [Parameter]
        x : 入力データ
        t : 正解ラベル
        
        [Return]
        grads : それぞれの重みの勾配
        """
        loss_W = lambda w : self.loss(x, t)
        
        grads = {}
        
        # ここで使用しているnumerical_gradient関数はcommonディレクトリに存在する関数であるため、再帰をしているわけではない
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"]) # (input_size, hidden_size)サイズの勾配を計算
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"]) # hidden_sizeサイズの勾配を計算
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"]) # (hidden_size, output_size)サイズの勾配を計算
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"]) # output_sizeサイズの勾配を計算
        
        return grads
        
        
        


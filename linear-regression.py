
import torch as t
from torch.autograd import Variable 
import torch.nn as nn 
import matplotlib.pyplot as plt
import argparse

# 生成随机数据
x = t.unsqueeze(t.linspace(-1,1,200),dim=1)
y = 5 * x + 0.8 * t.rand(x.size())

X = Variable(x)
Y = Variable(y)


def init_params():
    W = Variable(t.randn(1,1),requires_grad=True)
    b = Variable(t.zeros(1,1),requires_grad=True)
    return {"W":W,"b":b}

# linear model
def model(X,params):
    return X * params["W"] + params["b"]
# loss
def square_loss(y_pred,y_true):
    return (y_pred - y_true).pow(2).sum()

def update_params(params,lr):
    params["W"].data -= lr * params["W"].grad.data
    params["b"].data -= lr * params["b"].grad.data
    return 

# train
def train():
    epochs = 100
    lr = 1e-3
    
    params = init_params()
    for i in range(epochs):
        y_pred = model(X,params) # 进行预测

        loss = square_loss(y_pred,y)

        loss.backward() # 反向求导

        update_params(params,lr) # 梯度下降，更新参数

        if (i + 1) % 20 == 0:
            print(loss.data.item())

        # 自动求导，梯度会自动累积，需要手动清除
        params["W"].grad.data.zero_()
        params["b"].grad.data.zero_()

    plt.figure()
    plt.scatter(X.data.numpy(),Y.data.numpy())
    plt.plot(X.data.numpy(),y_pred.data.numpy(),"r-",lw=4)
    plt.show()

    print("The truh:5,predict parameter:{:.4f}".format(params["W"].data.item()))


def torch_nn_train():
    epochs = 100
    lr = 1e-3

    # 定义模型，使用torch.nn.model
    model = nn.Linear(1,1)

    # 定义损失函数
    mse = nn.MSELoss(reduction="sum")

    # 定义优化方法
    optimizer = t.optim.SGD(model.parameters(),lr=lr)

    for i in range(epochs):
        y_pred = model(X)

        loss = mse(y_pred,Y)

        if (i + 1) % 20 == 0:
            print(loss.data.item())

        optimizer.zero_grad() # 清空上一次的梯度

        loss.backward() # 反向求导

        optimizer.step() # 更新参数
    
    
    for k,v in model.state_dict().items():
        print("parameter name:{},predict value:{:.4f}".format(k,v.data.item()))

    plt.figure()
    plt.scatter(X.data.numpy(),Y.data.numpy())
    plt.plot(X.data.numpy(),y_pred.data.numpy(),"r-",lw=4)
    #plt.show()

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data","-d",default=True,help="Whether show the random data.")
    parser.add_argument("--method","-m",required=True,help="choose the method,'nn' or 'hand'")

    args = parser.parse_args()

    if args.data == True:
        plt.figure()
        plt.scatter(x.numpy(),y.numpy())
        #plt.show()

    if args.method == "nn":
        torch_nn_train()
    elif args.method == "hand":
        train()
    else:
        print("method must be 'nn' or 'hand'")
    
    plt.show()

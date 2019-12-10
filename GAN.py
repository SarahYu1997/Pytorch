import torch
from torch import nn, optim, autograd
import numpy as np
import visdom
import random  # 注意numpy里也有个random！！！
from matplotlib import pyplot as plt

h_dim = 400
batchsz = 512  # 一般是32、64，根据显卡来决定，这里数据量很少，所以就设大一点

viz = visdom.Visdom()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # 输入z:[b, 2],如果是图像的话，这个2通常是100
            # 输出[b, 2],这个2是学习的真实分布x的维度；这里用2是方便可视化这个真实的分布
            # [b, 2]->[b, 2]
            nn.Linear(2, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, 2)  # 最后一层线性层没有ReLU函数，因为不需要'单侧抑制'
        )

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # 输入x:[b, 2]隐藏变量2可以自定义
            # [b, 2]->[b, 1]
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),  # 输出判断输入为真(1)的‘概率’
            nn.Sigmoid()  # 将上述概率值更改到0-1范围内
        )

    def forward(self, x):
        output = self.net(x)  # x->[b, 1]
        return output.view(-1)  # x->[b]


def data_generator():
    # 总体符合由8个高斯分布组合而成的混合分布
    # 先设计分布函数，再从分布中sample出数据集
    scale = 2.  # 2.0
    centers = [(0, 1),
               (1, 0),
               (0, -1),
               (-1, 0),
               (1./np.sqrt(2), 1./np.sqrt(2)),  # 1./numpy.sqrt(2)->0.7071067811865475
               (1./np.sqrt(2), -1./np.sqrt(2)),
               (-1./np.sqrt(2), -1./np.sqrt(2)),
               (-1./np.sqrt(2), 1./np.sqrt(2))]  # 8个高斯分布的中心点

    centers = [(scale*x, scale*y) for x, y in centers]  # 放大

    # while做一个迭代器
    while True:
        dataset=[]

        # 生成一个batch_size大小的点集
        for i in range(batchsz):
            point = np.random.randn(2) * 0.02 # randn()从标准正态分布中sample一个值或多个值；point->[0.05056171 0.49995133]
            center = random.choice(centers)  # 从centers中随机选取一个中心点，torch的random
            # point~N(0,1)*0.02 -> 给point加一个variance
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset).astype(np.float32)  # 从列表创建ndarray;将ndarray中的数据类型转化为float32
        dataset /= 1.414  # 缩小
        yield dataset  # 返回值，停下，下次从这里开始调用


def generate_image(D, G, xr, epoch):
    """
    Generates and saves a plot of the true distribution, the generator, and the critic.
    """
    # 我自己修改部分
    xr = xr.cpu().numpy()

    N_POINTS = 128
    RANGE = 3
    plt.clf()
    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))
    # (16384, 2)
    # print('p:', points.shape)
    # draw contour
    with torch.no_grad():
        points = torch.Tensor(points).cuda() # [16384, 2]
        disc_map = D(points).cpu().numpy() # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)
    # plt.colorbar()
    # draw samples
    with torch.no_grad():
        z = torch.randn(batchsz, 2).cuda() # [b, 2]
        samples = G(z).cpu().numpy() # [b, 2]
    plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')
    viz.matplot(plt, win='contour', opts=dict(title='p(x):%d'%epoch))


def main():
    torch.manual_seed(23)  # 生成z的时候用到
    np.random.seed(23)  # data_generator用到；seed()用于指定随机数生成时所用算法开始的整数值，之后生成的随机数按顺序取值

    data_iter = data_generator()
    # x = next(data_iter)  # 检查data_generator好使不
    # print(x.shape)  # (512, 2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G = Generator().to(device)
    D = Discriminator().to(device)
    # print(next(G.parameters()).is_cuda)  # True;g.parameters()返回一个迭代器，不能直接输出
    # print(G)
    # print(D)

    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss',legend=['D', 'G']))

    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))  # 根据经验，betas就这么设置，有经验了再自己改
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))

    # 交替train G & D
    for epoch in range(50000):
        # 1.train Discrimator firstly
        # 可以是1次，可以是5次
        for _ in range(5):
            # 1.1train on real data
            xr = next(data_iter)  # x->ndarray类型
            xr = torch.from_numpy(xr).to(device)  # x->tensor类型
            predr = D(xr)  # [b, 2]->[b, 1]
            lossr = -predr.mean()  # 不指定参数，计算所有元素的均值

            # 1.2train on fake data
            z = torch.randn(batchsz, 2).to(device)
            xf = G(z).detach()  # tf.stop_gradient(),相当于关掉水闸，梯度从predf开始往反向算，算到xf停止；因为只需要更新D，算D的梯度就可以了
            predf = D(xf)
            lossf = predf.mean()

            loss_D = lossr+lossf

            # optimizer
            optim_D.zero_grad()  # D网路的梯度清零
            loss_D.backward()  # 向后传播，计算D网络的梯度，注意，G的闸门关了哦
            optim_D.step()  # 更新D的参数

        # 2.train Generator
        z = torch.randn(batchsz, 2).to(device)
        xf = G(z)
        predf = D(xf)  # 不能加detach，D的梯度算就算了，不更新就好，就是因为这一步产生了D的梯度信息，所以上面D更新参数时，一定要清零，不然会累加
        loss_G = -predf.mean()

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')

            print(loss_D.item(), loss_G.item())  # Returns the value of this tensor as a standard Python number. This only works for tensors with one element.

            generate_image(D, G, xr, epoch)


if __name__ == '__main__':
    main()

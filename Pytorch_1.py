# import torch
# print("hello pytorch{}".format(torch.__version__))
# print(torch.cuda.is_available())

from __future__ import print_function
import torch

# x = torch.empty(5, 3) # 构造一个未初始化的5行3列的张量
# print(x)
#
# y = torch.rand(5, 3) # 在区间[0,1)[0,1)上由均匀分布的随机数填充的张量
# print(y)
#
#
# z = torch.zeros(5, 3, dtype=torch.long)
# print(z)

# x = torch.tensor([5.5, 3]) # 用[5.5,3]构造张量 输出为[5.5,3]
# print(x)
#
# x = x.new_ones(5, 3, dtype=torch.double) # 构造5行3列的double类型的张量覆盖x以前的值
# print(x)
#
# x = torch.randn_like(x, dtype=torch.float)
# print(x)
# print(x.size())
#
# y = torch.rand(5, 3) # 构造5行3列用0-1的随机数填充的张量
# print(y)
# # 加法的三种形式
# print(x + y)
# print(torch.add(x, y))
#
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)
#
# # 将x加到y上
# y.add_(x)
# print(y)
# print(y[:, 1])

# 用torch.view改变形状，改变后的形状和原形状中大小相等
# x = torch.randn(4, 4) # 用0-1之间的随机数填充4行4列的张量
# y = x.view(16) # 将大小变为1行16列
# z = x.view(-1, 8) # 将大小变为2行8列
# print(x.size(), y.size(), z.size())
# 输出为torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
# print(x)
# print(y)
# print(z)
# 输出x，y，z中数值相同，形状不同


# x.item() 可以获得x的值
# x = torch.randn(1)
# print(x)
# print(x.item())


# requires_grad = True时，会跟踪针对tensor的所有操作
# x = torch.ones(2, 2, requires_grad=True) # torch.ones（）元素全为1
# print(x)
#
# y = x + 2
# print(y)
# print(y.grad_fn)
# z = y * y * 3
# out = z.mean()
#
# print(z, out)

# requires_grad 默认是false
# a = torch.randn(2, 2) # .randn表示用正态分布值填充
# print(a)
# a = ((a * 3)/(a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)


# y.data.norm 表示求y的范数
# torch.tensor 构造新的张量
x = torch.randn(3, requires_grad=True)
# print(x)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
# print(v)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():  # 停止对从跟踪历史中的.requires_grad=True的张量自动求导。
    print((x ** 2).requires_grad)


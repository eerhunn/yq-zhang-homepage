"""Reproduce the XOR experiment with PyTorch: python mlp.py."""  # 用 PyTorch 复现 XOR 实验，在命令行运行 python mlp.py。
import torch  # 导入 PyTorch，用于张量运算和自动求导。
from torch.nn import Parameter  # 导入 Parameter，将普通张量标记为可训练参数。
from torch.nn.functional import binary_cross_entropy_with_logits  # 导入直接接收原始分数的二元交叉熵函数，其内部计算比先求概率再取对数更稳定。


def main():  # 定义主函数，把数据准备、训练和结果输出组织在一起。
    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])  # 建立形状为 4×2 的输入张量，每行是一组 XOR 输入。
    targets = torch.tensor([[0.], [1.], [1.], [0.]])  # 建立形状为 4×1 的标签张量，输入不同为 1，相同为 0。
    weights = Parameter(torch.tensor([[2., -2.], [-2., 2.]]))  # 初始化 2×2 权重矩阵，行对应输入，列对应隐藏单元。
    hidden_bias = Parameter(torch.tensor([-1., -1.]))  # 初始化两个隐藏单元的偏置，计算加权和时会逐行广播。
    output_weights = Parameter(torch.tensor([[3.], [3.]]))  # 初始化 2×1 输出权重，将两个隐藏值组合成一个输出分数。
    output_bias = Parameter(torch.tensor([-2.5]))  # 初始化输出偏置，使每个样本的输出分数都加上这个值。
    optimizer = torch.optim.Adam(  # 开始创建 Adam 优化器，用来根据梯度更新参数。
        [weights, hidden_bias, output_weights, output_bias], lr=0.03  # 指定四组可训练参数以及学习率 0.03。
    )  # 结束优化器配置。

    for epoch in range(301):  # 依次计算第 0 至 300 轮的状态，第 0 轮是尚未更新参数的状态。
        hidden = torch.sigmoid(inputs @ weights + hidden_bias)  # 执行输入层到隐藏层的矩阵乘法，加偏置后经 sigmoid 得到 4×2 的隐藏值。
        logits = hidden @ output_weights + output_bias  # 组合隐藏值并加输出偏置，得到 4×1 原始分数 logits。
        loss = binary_cross_entropy_with_logits(logits, targets)  # 计算四个样本的平均二元交叉熵，此处直接输入 logits。
        if epoch in (0, 100, 200, 300):  # 仅在这四个轮次打印状态，减少输出量。
            probabilities = logits.detach().sigmoid().flatten()  # detach 切断打印数据与计算图的联系，sigmoid 将分数转为概率，flatten 展平。
            print(f"Epoch {epoch:3d} | loss {loss.item():.6f} | p(1) {probabilities.tolist()}")  # 输出当前轮次、平均损失以及每个输入属于类别 1 的概率。
        if epoch == 300:  # 检测是否已完成 300 次参数更新。
            break  # 结束循环，因此虽然记录 301 个状态，实际只更新参数 300 次。
        optimizer.zero_grad()  # 清除上一轮梯度，防止 PyTorch 默认的梯度累加影响本轮训练。
        loss.backward()  # 从损失反向传播，利用链式法则计算各参数的梯度。
        optimizer.step()  # 由 Adam 读取梯度并更新权重和偏置，进入下一个训练状态。

    for name, value in (  # 依次遍历参数名称和对应张量，准备输出训练后的数值。
        ("W", weights), ("b_hidden", hidden_bias),  # 为输入权重和隐藏偏置附上方便阅读的名称。
        ("v", output_weights), ("b_output", output_bias),  # 为输出权重和输出偏置附上方便阅读的名称。
    ):  # 结束参数列表，并开始处理其中每一项。
        print(f"{name} = {value.detach().tolist()}")  # 将参数从计算图分离并转为普通列表，打印最终学习结果。


if __name__ == "__main__":  # 仅在直接运行此文件时进入训练，被其他程序导入时不会自动执行。
    main()  # 调用主函数，启动完整实验。

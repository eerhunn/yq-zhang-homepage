"""Reproduce the XOR experiment with TensorFlow: python mlp_tensorflow.py."""  # 用 TensorFlow 复现 XOR 实验，在命令行运行 python mlp_tensorflow.py。
import tensorflow as tf  # 导入 TensorFlow，并使用常见缩写 tf。


def main():  # 定义主函数，统一组织数据、参数、训练和结果输出。
    inputs = tf.constant([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])  # 建立 4×2 的常量张量，每行包含两个 XOR 输入。
    targets = tf.constant([[0.], [1.], [1.], [0.]])  # 建立 4×1 的标签张量，输入不同为 1，相同为 0。
    weights = tf.Variable([[2., -2.], [-2., 2.]])  # 建立可训练的 2×2 输入权重矩阵，行对应输入，列对应隐藏单元。
    hidden_bias = tf.Variable([-1., -1.])  # 建立两个隐藏单元的可训练偏置。
    output_weights = tf.Variable([[3.], [3.]])  # 建立可训练的 2×1 输出权重矩阵。
    output_bias = tf.Variable([-2.5])  # 建立可训练的输出偏置，对所有样本广播相加。
    parameters = [weights, hidden_bias, output_weights, output_bias]  # 将四组可训练参数收集起来，后续统一求梯度和更新。
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)  # 创建 Adam 优化器并指定学习率 0.03，其默认 epsilon 可能与 PyTorch 不同。

    for epoch in range(301):  # 依次记录第 0 至 300 轮状态，第 0 轮表示尚未更新参数。
        with tf.GradientTape() as tape:  # 启动自动微分记录器，记录作用于可训练变量的运算以便反向求导。
            hidden = tf.sigmoid(tf.matmul(inputs, weights) + hidden_bias)  # 计算输入乘权重加隐藏偏置，再用 sigmoid 得到 4×2 隐藏值。
            logits = tf.matmul(hidden, output_weights) + output_bias  # 组合隐藏值并加输出偏置，得到每个样本的原始输出分数。
            loss = tf.reduce_mean(  # 开始对四个样本的损失求平均。
                tf.nn.sigmoid_cross_entropy_with_logits(  # 使用直接接收 logits 的稳定二元交叉熵函数。
                    labels=targets, logits=logits  # 指定真实标签和当前原始输出分数，二者形状均为 4×1。
                )  # 结束逐样本交叉熵的计算。
            )  # 结束求平均，得到可用于反向传播的标量损失。
        if epoch in (0, 100, 200, 300):  # 仅在指定轮次打印实验状态。
            probabilities = tf.sigmoid(logits).numpy().flatten().tolist()  # 将分数转换为概率，再转为 NumPy 数组并展平为普通列表。
            print(f"Epoch {epoch:3d} | loss {float(loss.numpy()):.6f} | p(1) {probabilities}")  # 打印轮次、平均损失和四个样本属于类别 1 的概率。
        if epoch == 300:  # 检查是否已经完成 300 次参数更新。
            break  # 结束循环，避免再更新一次参数。
        gradients = tape.gradient(loss, parameters)  # 用记录的运算通过链式法则计算损失对四组参数的梯度。
        optimizer.apply_gradients(zip(gradients, parameters))  # 将每个梯度与对应参数配对，再由 Adam 更新参数。

    for name, value in (  # 遍历四组参数，输出训练后学习到的最终数值。
        ("W", weights), ("b_hidden", hidden_bias),  # 为输入权重和隐藏偏置提供打印名称。
        ("v", output_weights), ("b_output", output_bias),  # 为输出权重和输出偏置提供打印名称。
    ):  # 结束参数列表，开始逐项打印。
        print(f"{name} = {value.numpy().tolist()}")  # 将 TensorFlow 变量转为普通数值列表，并显示参数名称。


if __name__ == "__main__":  # 只有直接运行本文件时才启动训练流程。
    main()  # 调用主函数，完成 XOR 实验。

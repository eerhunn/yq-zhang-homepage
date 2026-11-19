"""Reproduce the XOR experiment with TensorFlow: python mlp_tensorflow.py."""
import tensorflow as tf


def main():
    inputs = tf.constant([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = tf.constant([[0.], [1.], [1.], [0.]])
    weights = tf.Variable([[2., -2.], [-2., 2.]])
    hidden_bias = tf.Variable([-1., -1.])
    output_weights = tf.Variable([[3.], [3.]])
    output_bias = tf.Variable([-2.5])
    parameters = [weights, hidden_bias, output_weights, output_bias]
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)

    for epoch in range(301):
        with tf.GradientTape() as tape:
            hidden = tf.sigmoid(tf.matmul(inputs, weights) + hidden_bias)
            logits = tf.matmul(hidden, output_weights) + output_bias
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=targets, logits=logits
                )
            )
        if epoch in (0, 100, 200, 300):
            probabilities = tf.sigmoid(logits).numpy().flatten().tolist()
            print(f"Epoch {epoch:3d} | loss {float(loss.numpy()):.6f} | p(1) {probabilities}")
        if epoch == 300:
            break
        gradients = tape.gradient(loss, parameters)
        optimizer.apply_gradients(zip(gradients, parameters))

    for name, value in (
        ("W", weights), ("b_hidden", hidden_bias),
        ("v", output_weights), ("b_output", output_bias),
    ):
        print(f"{name} = {value.numpy().tolist()}")


if __name__ == "__main__":
    main()

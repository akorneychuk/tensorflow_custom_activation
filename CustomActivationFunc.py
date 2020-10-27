from tensorflow.python.framework import ops
import numpy as np
import tensorflow as tf
import os
import sys


print(tf.__version__)
print(sys.version)


def produce_initial_weights(shape):
    return (0.2 * np.random.random(shape) - 0.1).reshape(shape).astype('float32')


def leaky_relu(x):
    if (x < 0):
        return 0.01 * x
    if (x >= 0 and x <= 1):
        return x
    if (x > 1):
        return 1 + 0.01 * (x - 1)


def d_leaky_relu(x):
    if (x < 0):
        return 0.01
    if (x >= 0 and x <= 1):
        return 1
    if (x > 1):
        return 0.01


np_leaky_relu = np.vectorize(leaky_relu, otypes=[np.float32])
np_d_leaky_relu = np.vectorize(d_leaky_relu, otypes=[np.float32])

np_leaky_relu_32 = lambda x: np_leaky_relu(x).astype(np.float32)
np_d_leaky_relu_32 = lambda x: np_d_leaky_relu(x).astype(np.float32)


def relu_grad(op, grad):
    x = op.inputs[0]
    r = tf.mod(x, 1)
    n_gr = tf.to_float(tf.less_equal(r, 0.5))
    grad_res = grad * n_gr
    return grad_res


def py_func(func, inp, Tout, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+2))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.Graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_function(func, inp, Tout, name)


def tf_leaky_relu(x,name=None):
    with ops.name_scope(name, "d_spiky", [x]) as name:
        y = py_func(np_leaky_relu_32,  # forward pass function
                    [x],
                    [tf.float32],
                    name=name,
                    grad=relu_grad)  # the function that overrides gradient
        return y[0]


def tf_d_leaky_relu(x,name=None):
    with ops.name_scope(name, "d_leaky_relu", [x]) as name:
        y = py_func(np_d_leaky_relu_32,
                    [x],
                    [tf.float32],
                    name=name)
        return y[0]


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def mse(true, predicted):
        mse_res = tf.reduce_mean(tf.square(true - predicted))
        return mse_res

    hidden_layer_1_size = 12
    hidden_layer_2_size = 10

    np.random.seed(1)

    inputs = np.array([[1, 0, 1, 1, 0],
                       [0, 0, 1, 0, 1],
                       [0, 1, 1, 1, 1],
                       [1, 0, 0, 0, 0],
                       [1, 1, 0, 1, 0],
                       [0, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 1]]).astype('float32')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


    def mse(true, predicted):
        mse_res = tf.reduce_mean(tf.square(true - predicted))
        return mse_res


    hidden_layer_1_size = 12
    hidden_layer_2_size = 10

    np.random.seed(1)

    inputs = np.array([[1, 0, 1, 1, 0],
                       [0, 0, 1, 0, 1],
                       [0, 1, 1, 1, 1],
                       [1, 0, 0, 0, 0],
                       [1, 1, 0, 1, 0],
                       [0, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 1]]).astype('float32')

    weights_0_1 = tf.Variable(produce_initial_weights([inputs.shape[1], hidden_layer_1_size]))
    bias_0_1 = tf.Variable(produce_initial_weights([1, hidden_layer_1_size]))
    weights_1_2 = tf.Variable(produce_initial_weights([hidden_layer_1_size, hidden_layer_2_size]))
    bias_1_2 = tf.Variable(produce_initial_weights([1, hidden_layer_2_size]))
    weights_2_3 = tf.Variable(produce_initial_weights([hidden_layer_2_size, 1]))
    expects = tf.constant(np.array([1, 1, 0, 0, 1, 0, 0, 1]).astype('float32'))

    cached_weigts = []
    error_holder = []

    for j in range(3000):
        error, correct_cnt = (0.0, 0)
        full_iteration_error = 0
        for i in range(len(expects)):
            input = tf.reshape(tf.constant(inputs[i]), [1, 5])
            expect = tf.reshape(tf.constant(expects[i]), [1, 1])

            with tf.GradientTape(persistent=True) as t:
                t.watch(weights_0_1)
                t.watch(bias_0_1)
                t.watch(weights_1_2)
                t.watch(bias_1_2)
                t.watch(weights_2_3)

                # Does works as expected! :)
                # hidden_1 = tf.nn.relu(tf.add(tf.matmul(input, weights_0_1), bias_0_1))
                # hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weights_1_2), bias_1_2))
                # predict = tf.nn.relu(tf.matmul(hidden_2, weights_2_3))
                
                # Does not works..:(
                hidden_1 = tf_leaky_relu(tf.add(tf.matmul(input, weights_0_1), bias_0_1))
                hidden_2 = tf_leaky_relu(tf.add(tf.matmul(hidden_1, weights_1_2), bias_1_2))
                predict = tf_leaky_relu(tf.matmul(hidden_2, weights_2_3))

                

                loss = mse(predict, expect)

            grads = t.gradient(loss, [weights_0_1, bias_0_1, weights_1_2, bias_1_2, weights_2_3])
            optimizer = tf.optimizers.Adam(0.01)
            optimizer.apply_gradients(zip(grads, [weights_0_1, bias_0_1, weights_1_2, bias_1_2, weights_2_3]))

            error += np.sum(loss)
            correct_cnt += int(np.argmax(predict) == np.argmax(expect))

            full_iteration_error = np.sum((predict - expect) ** 2)

        weights_0_1_current = weights_0_1.value().numpy().copy()
        weights_1_2_current = weights_1_2.value().numpy().copy()

        if j % 100 == 0:
            cached_weigts.append({
                "error": error,
                "weigts": {
                    "weights_0_1": weights_0_1_current,
                    "weights_1_2": weights_1_2_current
                }
            })

            print("Error: ", error)
            print("Correct cnt: ", correct_cnt)

    print("END")
    print("cached_weigts: ", cached_weigts)

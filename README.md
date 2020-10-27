# Tensorflow Custom Activation Func
There is a simple pure TensorFlow multi-layer perceptron with custom activation function and derivative for gradient descent

**System information**
- MAC OS Catalina 10.15.7
- Python v3.8.5
- TensorFlow v2.3.1

**Stacktrace**

`Traceback (most recent call last):
  File "/private/var/root/PycharmProjects/untitled_sec/venv/lib/python3.8/site-packages/tensorflow/python/eager/backprop.py", line 162, in _gradient_function
    return grad_fn(mock_op, *out_grads)
  File "/private/var/root/PycharmProjects/untitled_sec/venv/lib/python3.8/site-packages/tensorflow/python/ops/math_grad.py", line 1694, in _MatMulGrad
    grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True)
  File "/private/var/root/PycharmProjects/untitled_sec/venv/lib/python3.8/site-packages/tensorflow/python/ops/gen_math_ops.py", line 5624, in mat_mul
    _ops.raise_from_not_ok_status(e, name)
  File "/private/var/root/PycharmProjects/untitled_sec/venv/lib/python3.8/site-packages/tensorflow/python/framework/ops.py", line 6843, in raise_from_not_ok_status
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError: In[0] is not a matrix. Instead it has shape [] [Op:MatMul]`

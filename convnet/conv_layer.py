import numpy as np
from conv2d import Conv2D
from maxpool2d import MaxPool2D


def relu(z):
    return np.where(z > 0, z, 0)


def main():

    input = np.random.rand(32, 3, 28, 28) - 0.5
    conv2d = Conv2D(3, 16, kernel_size=(5, 5), stride=1, padding="valid")
    maxpool2d = MaxPool2D(kernel_size=(2, 2), stride=2, padding="valid")
    conv_output = conv2d.forward(input)
    pool_output = maxpool2d.forward(conv_output)
    activation = relu(pool_output)
    assert activation.shape == (32, 16, 12, 12)
    assert np.min(activation) == 0
    assert np.max(activation) == np.max(pool_output)


if __name__ == "__main__":
    main()

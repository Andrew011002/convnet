import numpy as np


class Conv2D:

    def __init__(self, in_c, out_c, kernel_size, stride=1, padding="valid", fill=0):
        assert len(kernel_size) == 2, "can only convolve with 2D kernel"
        assert kernel_size[0] == kernel_size[1], "can only convolve with square kernels"
        assert stride >= 1, "can't implement stride smaller than 1"
        if type(padding) in [int, float]:
            padding = (padding, padding)
        if isinstance(padding, tuple):
            assert len(padding) == 2, "can only pad with tuple of 2 values"
            assert all(type(p) in [int, float] for p in padding), \
                "can only pad with numeric types"
        else:
            assert padding in ["valid", "same"], \
                "padding types can only be valid or same"
        assert type(fill) in [int, float], "can only fill with numeric types"

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.fill = fill
        self.in_c = in_c
        self.out_c = out_c
        self.weight = np.random.randn(
            out_c, in_c, kernel_size[0], kernel_size[1])
        self.bias = np.zeros((1, out_c, 1, 1))

    def forward(self, x):
        x = self.pad(x, self.weight, self.stride, self.padding, self.fill)
        return self.convolve2d(x, self.weight, self.stride) + self.bias

    def convolve2d(self, input, kernel, stride):
        batch_size, in_c, h, w = input.shape
        out_c, in_c, k, k = kernel.shape
        shape = (batch_size, out_c, (h - k) //
                 stride + 1, (w - k) // stride + 1)
        output = np.empty(shape)
        for w_k in range(out_c):
            for i in range(0, h - k + 1, stride):
                for j in range(0, w - k + 1, stride):
                    out = np.sum(input[:, :, i:i + k, j:j + k]
                                 * kernel[w_k], axis=(1, 2, 3))
                    output[:, w_k, i // stride, j // stride] = out
        return output

    def pad(self, input, kernel, stride=1, padding="valid", fill=0):
        batch_size, in_c, h, w = input.shape
        out_c, in_c, k, k = kernel.shape
        pad = self._get_pad_val(h, w, k, stride, padding)
        n, m = h + 2 * pad[0], w + 2 * pad[1]
        pad_arr = np.full((batch_size, in_c, n, m),
                          fill_value=fill, dtype=np.float32)
        pad_arr[:, :, pad[0]:pad[0] + h, pad[1]:pad[1] + w] = input
        return pad_arr

    def _get_pad_val(self, h, w, k, stride, padding):
        if padding == "valid":
            return (0, 0)
        elif padding == "same":
            p_h = np.ceil((k + h * (stride - 1) - stride) / 2)
            p_w = np.ceil((k + w * (stride - 1) - stride) / 2)
            return tuple(map(int, (p_h, p_w)))
        return padding


def main():
    input = np.random.randn(16, 3, 32, 32)
    conv2d = Conv2D(3, 3, kernel_size=(3, 3), stride=1, padding="same")
    output = conv2d.forward(input)
    assert input.shape == output.shape

    input = np.random.randn(16, 3, 55, 32)
    conv2d = Conv2D(3, 20, kernel_size=(3, 3), stride=1, padding="same")
    output = conv2d.forward(input)
    assert output.shape == (16, 20, 55, 32)

    input = np.random.randn(8, 3, 100, 100)
    conv2d = Conv2D(3, 32, kernel_size=(5, 5), stride=1, padding="valid")
    output = conv2d.forward(input)
    assert output.shape == (8, 32, 96, 96)

    input = np.random.randn(1, 3, 48, 71)
    conv2d = Conv2D(3, 10, kernel_size=(10, 10), stride=2, padding="valid")
    output = conv2d.forward(input)
    assert output.shape == (1, 10, 20, 31)

    input = np.random.randn(1, 3, 48, 71)
    conv2d = Conv2D(3, 10, kernel_size=(10, 10), stride=2, padding="same")
    output = conv2d.forward(input)
    assert output.shape == (1, 10, 48, 71)

    input = np.random.randn(5, 3, 30, 30)
    conv2d = Conv2D(3, 16, kernel_size=(3, 3), stride=2, padding="valid")
    output = conv2d.forward(input)
    assert output.shape == (5, 16, 14, 14)

    input = np.random.randn(2, 3, 24, 24)
    conv2d = Conv2D(3, 4, kernel_size=(4, 4), stride=1, padding=(2, 2))
    output = conv2d.forward(input)
    assert output.shape == (2, 4, 25, 25)

    input = np.random.randn(1, 3, 15, 15)
    conv2d = Conv2D(3, 6, kernel_size=(3, 3),
                    stride=1, padding="valid", fill=1)
    output = conv2d.forward(input)
    assert output.shape == (1, 6, 13, 13)

    input = np.random.randn(1, 3, 32, 32)
    conv2d = Conv2D(3, 1, kernel_size=(3, 3), stride=2, padding="same")
    output = conv2d.forward(input)
    assert output.shape == (1, 1, 32, 32)

    input = np.random.randn(1, 3, 32, 32)
    conv2d = Conv2D(3, 1, kernel_size=(3, 3), stride=2, padding=(3, 3))
    output = conv2d.forward(input)
    assert output.shape == (1, 1, 18, 18)


if __name__ == "__main__":
    main()

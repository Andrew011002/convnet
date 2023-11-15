import numpy as np


class Conv2D:

    def __init__(self, kernel_size, stride=1, padding="valid", fill=0):
        assert len(kernel_size) == 2, "can only convolve with 2D kernel"
        assert kernel_size[0] == kernel_size[1], "can only convolve with square kernels"
        assert stride >= 1, "can't implement stride smaller than 1"
        if type(padding) in [int, float]:
            padding = (padding, padding)
        if isinstance(padding, tuple):
            assert len(padding) == 2, "can only pad with tuple of 2 values"
            assert all(p in [int, float] for p in padding), \
                "can only pad with numeric types"
        else:
            assert padding in ["valid", "same"], \
                "padding types can only be valid or same"
        assert type(fill) in [int, float], "can only fill with numeric types"

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.fill = fill
        self.weight = np.random.randn(*kernel_size)

    def forward(self, x):
        x = self.pad(x, self.weight, self.stride, self.padding, self.fill)
        return self.convovle2d(x, self.weight, self.stride)

    def convovle2d(self, input, kernel, stride):
        h, w = input.shape
        k = kernel.shape[0]
        shape = ((h - k) // stride + 1, (w - k) // stride + 1)
        output = np.empty(shape)
        for i in range(0, h - k + 1, stride):
            for j in range(0, w - k + 1, stride):
                out = np.sum(input[i: i + k, j: j + k] * kernel)
                output[i // stride, j // stride] = out
        return output

    def pad(self, input, kernel, stride=1, padding="valid", fill=0):
        h, w = input.shape
        k = kernel.shape[0]
        pad = self._get_pad_val(h, w, k, stride, padding)
        n, m = h + 2 * pad[0], w + 2 * pad[1]
        pad_arr = np.full((n, m), fill_value=fill, dtype=np.float32)
        pad_arr[pad[0]:pad[0] + h, pad[1]:pad[1] + w] = input
        return pad_arr

    def _get_pad_val(self, h, w, k, stride, padding):
        if padding == "valid":
            return (0, 0)
        elif padding == "same":
            p_h = (k + h * (1 - stride) - stride) // 2
            p_w = (k + w * (1 - stride) - stride) // 2
            return (p_h, p_w)
        return padding


def output_shape(image, kernel, pad, stride):
    h, w = image.shape
    k = kernel.shape[0]
    n, m = (h + 2 * pad[0] - k) // stride + 1, \
        (w + 2 * pad[1] - k) // stride + 1
    return n, m


def main():
    input = np.random.randn(512, 512)
    conv2d = Conv2D((3, 3), stride=1, padding="same")
    conv2d.weight = np.zeros_like(conv2d.weight)
    conv2d.weight[1, 1] = 1.0
    output = conv2d.forward(input)
    assert np.allclose(input, output)


if __name__ == "__main__":
    main()

import numpy as np


class Conv2D:

    def __init__(self, kernel_size, stride=1, padding="valid", fill=0):
        assert kernel_size[0] == kernel_size[1], "can only convolve with square kernels"
        assert stride >= 1, "can't implement stride smaller than 1"
        if isinstance(padding, int):
            assert padding >= 0, "padding must be larger than zero"
        else:
            assert padding in ["valid", "same"], \
                "padding types can only be valid or same"
        assert type(fill) in [int, float], "can only fill with numeric types"
        self.kernel_size = kernel_size
        self.stide = stride
        self.padding = padding
        self.fill = fill
        self.weight = np.random.randn(*kernel_size)

    def forward(self, x):
        pass

    def _convovle_2d(self, input, kernel, stride, padding, fill):
        pass

    def pad(self, input, kernel, stride=1, padding="valid", fill=0):
        n, k = input.shape[0], kernel.shape[0]
        p = self._get_pad_val(n, k, stride, padding)
        m = n + 2 * p
        pad_arr = np.full((m, m), fill_value=fill)
        pad_arr[p: n + p, p: n + p] = input
        return pad_arr

    def _get_pad_val(self, n, k, stride, padding):
        if padding == "same":
            pad = (k + n * (1 - stride) - stride) // 2
        elif padding == "valid":
            pad = 0
        else:
            pad = pad
        return pad


def main():
    conv2d = Conv2D((3, 3))
    input = np.random.rand(9, 9)
    output = conv2d.pad(input, conv2d.weight, stride=1, padding="same")
    assert output.shape == (11, 11)


if __name__ == "__main__":
    main()

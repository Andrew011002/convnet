import numpy as np


class MaxPool2D:

    def __init__(self, in_channels, kernel_size, stride=1, padding="valid", fill=0) -> None:
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
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.fill = fill
        self.kernel_size = (in_channels, kernel_size[0], kernel_size[1])

    def forward(self, x):
        x = self.pad(x, self.kernel_size, self.stride, self.padding, self.fill)
        return self.pool(x, self.kernel_size, self.stride)

    def pool(self, input, kernel_size, stride):
        batch_size, in_c, h, w = input.shape
        in_c, k, k = kernel_size
        n, m = (h - k) // stride + 1, (w - k) // stride + 1
        output = np.empty((batch_size, in_c, n, m))

        for i in range(0, h - k + 1, stride):
            for j in range(0, w - k + 1, stride):
                out = np.max(input[:, :, i:i + k, j:j + k], axis=(2, 3))
                output[:, :, i // stride, j // stride] = out
        return output

    def pad(self, input, kernel_size, stride=1, padding="valid", fill=0):
        batch_size, in_c, h, w = input.shape
        in_c, k, k = kernel_size
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
    pool = MaxPool2D(3, kernel_size=(3, 3), stride=1, padding="valid")
    input = np.random.rand(5, 3, 32, 32)
    output = pool.forward(input)
    print(output.shape)


if __name__ == "__main__":
    main()

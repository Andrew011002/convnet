import numpy as np


class Conv3D:

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="valid", fill=0):
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
        self.weight = np.random.randn(
            kernel_size[0], kernel_size[1], in_channels, out_channels)
        self.bias = np.zeros((1, 1, out_channels))

    def forward(self, x):
        x = self.pad(x, self.weight[:, :, :, 0],
                     self.stride, self.padding, self.fill)
        return self.convovle3d(x, self.weight, self.stride) + self.bias

    def convovle3d(self, input, kernel, stride):
        b, h, w, c = input.shape
        k, _, c, n_kernels = kernel.shape
        shape = (b, (h - k) // stride + 1, (w - k) //
                 stride + 1, n_kernels)
        output = np.empty(shape)
        for out_c in range(n_kernels):
            for i in range(0, h - k + 1, stride):
                for j in range(0, w - k + 1, stride):
                    out = np.sum(input[:, i: i + k, j: j + k]
                                 * self.weight[:, :, :, out_c], axis=(1, 2, 3))
                    output[:, i // stride, j // stride, out_c] = out
        return output

    def pad(self, input, kernel, stride=1, padding="valid", fill=0):
        b, h, w, c = input.shape
        k, _, c = kernel.shape
        pad = self._get_pad_val(h, w, k, stride, padding)
        n, m = h + 2 * pad[0], w + 2 * pad[1]
        pad_arr = np.full((b, n, m, c), fill_value=fill, dtype=np.float32)
        pad_arr[:, pad[0]:pad[0] + h, pad[1]:pad[1] + w, :] = input
        return pad_arr

    def _get_pad_val(self, h, w, k, stride, padding):
        if padding == "valid":
            return (0, 0)
        elif padding == "same":
            p_h = (k + h * (1 - stride) - stride) // 2
            p_w = (k + w * (1 - stride) - stride) // 2
            return (p_h, p_w)
        return padding


def main():
    conv3d = Conv3D(3, 3, kernel_size=(3, 3), padding="same")
    conv3d.weight = np.zeros_like(conv3d.weight)
    conv3d.weight[1, 1, :, :] = 1
    input = np.random.randn(32, 9, 9, 3)
    output = conv3d.forward(input)
    print(output.shape)
    assert np.allclose(input, output)


if __name__ == "__main__":
    main()

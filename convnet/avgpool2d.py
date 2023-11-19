import numpy as np


class AvgPool2D:

    def __init__(self, kernel_size, stride=1, padding="valid", fill=0) -> None:
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
        self.stride = stride
        self.padding = padding
        self.fill = fill
        self.kernel_size = (kernel_size[0], kernel_size[1])

    def forward(self, x):
        x = self.pad(x, self.kernel_size, self.stride, self.padding, self.fill)
        return self.pool(x, self.kernel_size, self.stride)

    def pool(self, input, kernel_size, stride):
        batch_size, in_c, h, w = input.shape
        k, k = kernel_size
        n, m = (h - k) // stride + 1, (w - k) // stride + 1
        output = np.empty((batch_size, in_c, n, m))

        for i in range(0, h - k + 1, stride):
            for j in range(0, w - k + 1, stride):
                out = np.mean(input[:, :, i:i + k, j:j + k], axis=(2, 3))
                output[:, :, i // stride, j // stride] = out
        return output

    def pad(self, input, kernel_size, stride=1, padding="valid", fill=0):
        batch_size, in_c, h, w = input.shape
        k, k = kernel_size
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


def tester(pool, input, stride):
    k, k = pool.kernel_size
    batch_size, c_in, old_h, old_w = input.shape
    h, w = (old_h - k) // stride + 1, (old_w - k) // stride + 1
    output = pool.forward(input)
    for b in range(batch_size):
        for c in range(c_in):
            for i in range(h):
                for j in range(w):
                    expected_max = np.mean(
                        input[b, c, i * stride:i * stride + k, j * stride:j * stride + k])
                    if not np.isclose(
                            output[b, c, i, j], expected_max):
                        return False
    return True


def main():
    pool = AvgPool2D(kernel_size=(3, 3), stride=2, padding="valid")
    input = np.round(np.random.rand(5, 3, 7, 7) * 10)
    output = pool.forward(input)
    assert output.shape == (5, 3, 3, 3)
    assert tester(pool, input, stride=pool.stride)

    pool = AvgPool2D(kernel_size=(2, 2), stride=2, padding="valid")
    input = np.random.rand(1, 3, 4, 4)
    output = pool.forward(input)
    assert output.shape == (1, 3, 2, 2)
    assert tester(pool, input, stride=pool.stride)

    pool = AvgPool2D(kernel_size=(3, 3), stride=1, padding="same")
    input = np.random.rand(2, 3, 5, 5)
    output = pool.forward(input)
    assert output.shape == (2, 3, 5, 5)

    pool = AvgPool2D(kernel_size=(2, 2), stride=3, padding="valid")
    input = np.random.rand(1, 3, 6, 6)
    output = pool.forward(input)
    assert output.shape == (1, 3, 2, 2)
    assert tester(pool, input, stride=pool.stride)

    pool = AvgPool2D(kernel_size=(3, 3), stride=2, padding=(1, 1), fill=-1)
    input = np.random.rand(1, 3, 4, 4)
    output = pool.forward(input)
    assert output.shape == (1, 3, 2, 2)


if __name__ == "__main__":
    main()

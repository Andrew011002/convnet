import numpy as np


def conv3d(image, kernel, stride=1, padding="same", fill=None):
    image = handle_padding(image, kernel.shape[0], stride, padding, fill)
    h, w, c = image.shape
    k = kernel.shape[0]

    out = np.zeros(((h - k) // stride + 1, (w - k) // stride + 1))
    for i in range(0, h - k + 1, stride):
        for j in range(0, w - k + 1, stride):
            out[i // stride, j // stride] = \
                np.sum(image[i:i + k, j:j + k, :] * kernel)
    return out


def handle_padding(image, k, stride, padding, fill):
    # assumes square image
    if padding == "same":
        p = (k - image.shape[0] * (1 - stride) - stride) // 2
    elif padding == "valid":
        p = 0
    elif isinstance(padding, int):
        p = padding
    fill = 0 if fill is None else fill
    image = pad(image, p, fill)
    return image


def pad(image, p=1, fill=0):
    if p == 0:
        return image
    h, w, c = image.shape
    n, m = h + 2 * p, w + 2 * p
    padded = np.full((n, m, c), fill_value=float(fill))
    padded[p:h + p, p:w + p, :] = image
    return padded


def main():
    image = np.random.randn(5, 5, 3)
    kernel = np.zeros((3, 3, 3))
    kernel[1, 1, :] = 1
    convd = conv3d(image, kernel, stride=1, padding="valid")
    assert (convd[0, 0] == np.sum(image[1, 1, :]))

    convd = conv3d(image, kernel, stride=1, padding="same", fill=0)
    assert np.array_equal(image.sum(axis=2), convd)

    convd = conv3d(image, kernel, stride=2, padding="valid")
    assert image[1, 1].sum() == convd[0, 0]


if __name__ == "__main__":
    main()

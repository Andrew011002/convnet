import numpy as np


def conv2d(image, kernel, padding="valid", fill=None):
    image = handle_padding(image, kernel.shape[0], padding, fill)
    h, w = image.shape
    k = kernel.shape[0]
    out = np.zeros((h - k + 1, w - k + 1))
    for i in range(h - k + 1):
        for j in range(w - k + 1):
            out[i, j] = np.sum(image[i:i + k, j:j + k] * kernel)
    return out


def handle_padding(image, k, padding, fill):
    if padding == "same":
        p = (k - 1) // 2
        fill = 0 if fill is None else fill
        image = pad(image, p, fill)
    return image


def pad(image, p=1, fill=0):
    h, w = image.shape
    n = h + 2 * p
    padded = np.full((n, n), fill_value=fill)
    padded[p:h + p, p:w + p] = image
    print(padded)
    return padded


def main():
    image = np.random.randint(0, 10, (5, 5))
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1
    out = conv2d(image, kernel, padding="same")
    assert image.shape == out.shape
    assert np.array_equal(image, out)

    out = conv2d(image, kernel, padding="valid")
    assert out.shape == (3, 3)

    out = conv2d(image, kernel, padding="same", fill=10)
    assert image.shape == out.shape
    # identity always results in identity regardless of pad fill
    assert np.array_equal(image, out)


if __name__ == "__main__":
    main()

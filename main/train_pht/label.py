import numpy as np


def generate_label(shape, idx, label_shape, label_width):
    # target = np.zeros(self.Y_shape, dtype=self.dtype)
    target = np.zeros(shape)
    idx = int(eval(idx))
    if label_shape == "gaussian":
        label_window = np.exp(
            -(np.arange(-label_width // 2, label_width // 2 + 1)) ** 2 / (2 * (label_width / 5) ** 2))
    elif label_shape == "triangle":
        label_window = 1 - np.abs(2 / label_width * (np.arange(-label_width // 2, label_width // 2 + 1)))
    else:
        print(f"Label shape {label_shape} should be guassian or triangle")
        raise

    if (idx - label_width // 2 >= 0) and (idx + label_width // 2 + 1 <= target.shape[0]):
        target[idx - label_width // 2:idx + label_width // 2 + 1] = label_window
    return target


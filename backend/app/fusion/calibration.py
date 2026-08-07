"""Temperature scaling applied to the fusion head's logits before thresholding."""


def calibrate(logits, temperature: float = 1.0):
    raise NotImplementedError

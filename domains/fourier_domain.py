import numpy as np
from PIL import Image

from domains.base_domain import BaseDomain


class FourierDomain(BaseDomain):
    name = "fourier"

    def transform(self, image):
        image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
        transformed_channels = []

        for channel_idx in range(image_array.shape[2]):
            channel = image_array[:, :, channel_idx]
            fft = np.fft.fft2(channel)
            fft_shifted = np.fft.fftshift(fft)
            log_magnitude = np.log1p(np.abs(fft_shifted))
            transformed_channels.append(log_magnitude)

        transformed = np.stack(transformed_channels, axis=-1)
        transformed -= transformed.min()
        max_value = transformed.max()
        if max_value > 0:
            transformed /= max_value

        transformed = (transformed * 255.0).astype(np.uint8)
        return Image.fromarray(transformed, mode="RGB")

import numpy as np
from PIL import Image

from domains.base_domain import BaseDomain


class WaveletDomain(BaseDomain):
    name = "wavelet"

    @staticmethod
    def _haar_subbands(channel):
        channel = channel.astype(np.float32)

        top_left = channel[0::2, 0::2]
        top_right = channel[0::2, 1::2]
        bottom_left = channel[1::2, 0::2]
        bottom_right = channel[1::2, 1::2]

        ll = (top_left + top_right + bottom_left + bottom_right) / 4.0
        lh = (top_left - top_right + bottom_left - bottom_right) / 4.0
        hl = (top_left + top_right - bottom_left - bottom_right) / 4.0
        hh = (top_left - top_right - bottom_left + bottom_right) / 4.0

        return ll, lh, hl, hh

    def transform(self, image):
        image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
        transformed_channels = []

        for channel_idx in range(image_array.shape[2]):
            channel = image_array[:, :, channel_idx]
            ll, lh, hl, hh = self._haar_subbands(channel)

            band_image = np.block([
                [ll, lh],
                [hl, hh],
            ])

            band_image -= band_image.min()
            max_value = band_image.max()
            if max_value > 0:
                band_image /= max_value

            transformed_channels.append(band_image)

        transformed = np.stack(transformed_channels, axis=-1)
        transformed = (transformed * 255.0).astype(np.uint8)
        return Image.fromarray(transformed, mode="RGB")

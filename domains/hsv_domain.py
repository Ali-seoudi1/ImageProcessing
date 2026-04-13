import numpy as np
from PIL import Image

from domains.base_domain import BaseDomain


class HSVDomain(BaseDomain):
    name = "hsv"

    def transform(self, image):
        hsv_image = image.convert("HSV")
        hsv_array = np.asarray(hsv_image, dtype=np.uint8)
        return Image.fromarray(hsv_array, mode="RGB")

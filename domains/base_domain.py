class BaseDomain:
    name = "spatial"

    def transform(self, image):
        return image.convert("RGB")

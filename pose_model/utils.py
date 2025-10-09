from PIL import Image


def load_png(path) -> Image:
    """
    Loads an PNG image replacing the alpha background with a neutral gray.
    Result is returned in RGB
    """
    image = Image.open(path).convert('RGBA')
    bg = Image.new('RGB', image.size, (128, 128, 128))
    image_rgb = Image.alpha_composite(bg.convert('RGBA'), image).convert('RGB')
    return image_rgb
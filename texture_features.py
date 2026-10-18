import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor


def load_grayscale(image_source):
    """Load an image from a path or file-like object."""
    image = Image.open(image_source).convert("L")
    image.thumbnail((512, 256))
    array = np.asarray(image, dtype=np.uint8)
    if array.size == 0:
        raise ValueError("The uploaded image is empty.")
    return array


def extract_texture_features(image_source):
    """Return a fixed-length feature vector: GLCM + LBP + Gabor."""
    gray = load_grayscale(image_source)
    quantized = (gray // 8).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )

    features = []
    for prop in [
        "contrast", "dissimilarity", "homogeneity",
        "energy", "correlation", "ASM"
    ]:
        values = graycoprops(glcm, prop)
        features.extend([float(np.mean(values)), float(np.std(values))])

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype(np.float64)
    lbp_hist /= max(lbp_hist.sum(), 1.0)
    features.extend(lbp_hist.tolist())

    gray_float = gray.astype(np.float32) / 255.0
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        real, imag = gabor(gray_float, frequency=0.2, theta=theta)
        magnitude = np.sqrt(real ** 2 + imag ** 2)
        features.extend([float(np.mean(magnitude)), float(np.std(magnitude))])

    result = np.asarray(features, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("Feature extraction produced invalid values.")
    return result


def display_grayscale(image_source):
    return Image.fromarray(load_grayscale(image_source))

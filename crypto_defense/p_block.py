import numpy as np
import scipy as scipy
from scipy import ndimage
from PIL import Image


def p(x: np.array, k: np.array)-> np.array:
    """
    permutes the image by sorting the input array k
    :param x: input image should be 3 dimensional, channel last
    :param k: values for a permutation matrix s, should be 2 dimensional 
    :return: 
    """

    k_flat = np.reshape(k, [-1])
    indices = np.argsort(k_flat)
    x_flat = np.reshape(x, [-1, x.shape[2]])
    x_scrambled_flat = x_flat[indices]
    x_scrambled = np.reshape(x_scrambled_flat, [x.shape[0], x.shape[1], x.shape[2]])

    return x_scrambled


def p_neighbors(x:np.array, k:np.array, patch_size:int=2) -> np.array:
    """
    Performas permutation specified by matrix of random values of k on patches of size patch_size * patch_size
    :param x:
    :param k:
    :param patch_size:
    :return:
    """
    scrambled_img = x.copy()
    w, h, _ = scrambled_img.shape
    for w_ind in range(0, w, patch_size):
        for h_ind in range(0, h, patch_size):
            key_excerpt = k[w_ind: w_ind + patch_size, h_ind: h_ind + patch_size]
            image_excerpt = scrambled_img[w_ind: w_ind + patch_size, h_ind: h_ind + patch_size]

            # randomize!
            key_excerpt_flat = np.reshape(key_excerpt, [-1])
            image_excerpt_flat = np.reshape(image_excerpt, [-1, scrambled_img.shape[-1]])
            indices = np.argsort(key_excerpt_flat)
            image_scrambled_flat = image_excerpt_flat[indices]
            image_scrambled = np.reshape(image_scrambled_flat, [patch_size, patch_size, scrambled_img.shape[-1]])
            scrambled_img[w_ind: w_ind + patch_size, h_ind: h_ind + patch_size, :] = image_scrambled
    return scrambled_img


def identity_key(x: np.array) -> np.array:
    k = np.array(range(x.shape[0] * x.shape[1]))
    return k


def inverse_key(x: np.array) -> np.array:
    k = np.array(range(x.shape[0] * x.shape[1], 0, -1))
    return k


def random_key(x: np.array) -> np.array:
    h, w, _ = x.shape
    k = np.random.rand(h, w)
    return k


if __name__ == "__main__":
    fname = "/home/hack/Downloads/guacamole.jpg"
    img = scipy.ndimage.imread(fname)

    key = random_key(img)
    # key = identity_key(img)
    # key = inverse_key(img)
    # key = local_random_key(img, patch_size=2)

    scrambled_img = p_neighbors(img, key, patch_size=100)
    Image.fromarray(img).show()
    Image.fromarray(scrambled_img).show()



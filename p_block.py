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

    # indices = np.argsort(k, axis=[0,1])
    k_flat = np.reshape(k, [-1])
    indices = np.argsort(k_flat)
    x_flat = np.reshape(x, [-1, x.shape[2]])
    x_scrambled_flat = x_flat[indices]
    x_scrambled = np.reshape(x_scrambled_flat, [x.shape[0], x.shape[1], x.shape[2]])
    # indices = np.dstack(np.unravel_index(np.argsort(k.ravel()), (k.shape[0], k.shape[1])))
    # scrambled_image = x[indices[0], indices[1], :]

    return x_scrambled

def p_neighbors(x:np.array, k:np.array, patch_size:int=2) -> np.array:
    k_flat = np.reshape(k, [-1])

    # go through half the array with stepsize patchsize / 2
    indices = [] * k_flat.shape[0]
    for i in range(0, x.shape[0] // 2, patch_size // 2):

        half_patchsize = 0
        first_cutout = k_flat[i: i + patch_size // 2]

        second_cutout = k_flat[i + x.shape[0] // 2: i + patch_size // 2 + x.shape[0] // 2]

        cutout = first_cutout + second_cutout
        indices_base_0 = np.argsort(cutout)
        indices[i: i + patch_size] = i + indices_base_0[0:patch_size // 2]
        indices[i + x.shape[0]//2: i + x.shape[0]//2 + patch_size // 2] = \
            x.shape[0] + i + indices_base_0[i + x.shape[0] // 2: i + x.shape[0] // 2 + patch_size // 2]
    x_flat = np.reshape(x, [-1, x.shape[2]])
    x_scrambled_flat = x_flat[indices]
    x_scrambled = np.reshape(x_scrambled_flat, [x.shape[0], x.shape[1], x.shape[2]])
    return x_scrambled

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

def local_random_key(x: np.array, patch_size:int=2) -> np.array:
    """
    We only randomize local patches, not to destroy the local pixel are similar assumption
    :param x:
    :return:
    """
    h, w, _ = x.shape
    # make sure that the patch_size * patch_size image patches stay where they are
    # crude heuristic should be done smarter

    key = np.zeros_like(x[:, :, 0], dtype=np.float32)
    for w_ind in range(0, w, patch_size):
        for h_ind in range(0, h, patch_size):
            vals = np.random.rand(patch_size, patch_size)
            for i in range(w_ind, w_ind + patch_size):
                for j in range(h_ind, h_ind + patch_size):
                    val_i = i - w_ind
                    val_j = j - h_ind
                    offset = (w_ind // patch_size * w + h_ind)
                    # print(offset)
                    key[i, j] = offset + vals[val_i, val_j]

            # for i in range(patch_size):
            #     for j in range(patch_size):
            #         ww = w_ind + i
            #         hh = h_ind + j
            #         offset = w_ind // patch_size * w + h_ind
            #         key[ww, hh] = offset + vals[i * patch_size + j]
    return key

    #w_offset = w // patch_size
    # h_offset = 500 #w_offset * w_offset
    # key = []
    # for w_ind in range(0, w, patch_size):
    #     w_ind_keys = []
    #     for h_ind in range(0, h, patch_size):
    #         w_ind_keys.extend(w * w_ind // patch_size + w_ind + np.random.rand(patch_size, patch_size))
    #     key.append(w_ind_keys)
    # key = np.reshape(key, [w, h])
    return key



if __name__ == "__main__":
    fname = "/home/hack/Downloads/guacamole.jpg"
    img = scipy.ndimage.imread(fname)

    # h, w, c = img.shape
    key = random_key(img)
    # key = identity_key(img)
    # key = inverse_key(img)
    # key = local_random_key(img, patch_size=2)

    # scrambled_img = p(img, key)
    scrambled_img = p_neighbors(img, key, patch_size=2)
    Image.fromarray(img).show()
    Image.fromarray(scrambled_img).show()



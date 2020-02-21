import random
import math
import numbers
import collections
import numpy as np
import torch
import cv2
import scipy.ndimage
import threading

from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            try:
                t.randomize_parameters()
            except AttributeError:
                pass

class MultiScaleRandomCrop(object):

    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        crop_size_x = int(image_width * self.scale)
        crop_size_y = int(image_height * self.scale)

        x1 = self.tl_x * (image_width - crop_size_x)
        y1 = self.tl_y * (image_height - crop_size_y)
        x2 = x1 + crop_size_x
        y2 = y1 + crop_size_y

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size[1], self.size[0]), self.interpolation) # Match torch's / torchvisions (h, w) convention

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.tl_x = random.random()
        self.tl_y = random.random()

class SpatialElasticDisplacement(object):

    def __init__(self, sigma=2.0, alpha=1.0, order=0, cval=0, mode="constant"):
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.cval = cval
        self.mode = mode

    def __call__(self, img):
        if self.p < 0.50:
            is_L = False
            is_PIL = isinstance(img, Image.Image)
            
            if is_PIL:
                img = np.asarray(img, dtype=np.uint8)
            if len(img.shape) == 2:
                is_L = True
                img = np.reshape(img, img.shape + (1,))  

            image = img
            image_first_channel = np.squeeze(image[..., 0])
            indices_x, indices_y = self._generate_indices(image_first_channel.shape, alpha=self.alpha, sigma=self.sigma)
            ret_image = (self._map_coordinates(
                image,
                indices_x,
                indices_y,
                order=self.order,
                cval=self.cval,
                mode=self.mode))

            if  is_PIL:
                if is_L:
                    return Image.fromarray(ret_image.reshape(ret_image.shape[:2]), mode= 'L')
                else:
                    return Image.fromarray(ret_image)
            else:
                return ret_image
        else:
            return img

    def _generate_indices(self, shape, alpha, sigma):
        assert (len(shape) == 2),"shape: Should be of size 2!"
        dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    def _map_coordinates(self, image, indices_x, indices_y, order=1, cval=0, mode="constant"):
        assert (len(image.shape) == 3),"image.shape: Should be of size 3!"
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in range(image.shape[2]):
            remapped_flat = scipy.ndimage.interpolation.map_coordinates(
                image[..., c],
                (indices_x, indices_y),
                order=order,
                cval=cval,
                mode=mode
            )
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result

    def randomize_parameters(self):
       self.p = random.random()

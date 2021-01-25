from imageio import imread
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

MAX_SEGMENT = 255


# from ex1 read image:
def read_image(filename, representation):
    """
    The next lines preform a image read to a matrix of numpy.float64 using
    imagio and numpy libraries.
    :param filename: a path to jpg image we would like to read.
    :param representation: 1 stands for grayscale , 2 for RGB.
    :return: image_mat - a numpy array represents the photo as described above.
    """
    image = imread(filename)
    if representation == 1:
        image_mat = np.array(rgb2gray(image))
    else:
        image_mat = np.array(image.astype(np.float64))
        image_mat /= MAX_SEGMENT
    return image_mat


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    A generator for generating pairs of image patches, corrupted and original
    :param filenames: a list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy array representation of an image as a single argument, and returning a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return:outputs random tuples of the form (source_batch, target_batch), where each output variable is an array of shape(batch_size, height, width, 1).
     target_batch is made of clean images and source_batch is their respective randomly corrupted version
     according to corruption_func(im)
    """
    my_image_dictionary = dict()
    while True:
        source_batch = []
        target_batch = []
        for i in range(batch_size):
            random_index = np.random.randint(0, len(filenames))
            random_im_name = filenames[random_index]
            random_patch_location = np.zeros(2).astype(np.int16)

            if random_im_name in my_image_dictionary:
                source_im_clipped = my_image_dictionary[random_im_name]
            else:
                source_im_clipped = read_image(random_im_name, 1)
                my_image_dictionary[random_im_name] = source_im_clipped

            big_crop = (crop_size[0] * 3, crop_size[1] * 3)
            random_patch_location[0] = np.random.randint(0 ,source_im_clipped.shape[0] - big_crop[0])
            random_patch_location[1] = np.random.randint(0 ,source_im_clipped.shape[1] - big_crop[1])

            patch_big = source_im_clipped[random_patch_location[0]:(random_patch_location[0] + big_crop[0]),
                                          random_patch_location[1]:(random_patch_location[1] + big_crop[1])]
            corrupted_im_patch_big = corruption_func(patch_big)

            random_patch_location[0] = np.random.randint(0, patch_big.shape[0] - crop_size[0])
            random_patch_location[1] = np.random.randint(0, patch_big.shape[1] - crop_size[1])

            patch = patch_big[random_patch_location[0]:(random_patch_location[0] + crop_size[0]),
                              random_patch_location[1]:(random_patch_location[1] + crop_size[1])]
            corrupted_im_patch = corrupted_im_patch_big[
                                 random_patch_location[0]:(random_patch_location[0] + crop_size[0]),
                                 random_patch_location[1]:(random_patch_location[1] + crop_size[1])]

            patch -= 0.5
            corrupted_im_patch -= 0.5

            source_batch.append(corrupted_im_patch)
            target_batch.append(patch)

        yield (source_batch, target_batch)


def noisy(image):
    row, col = image.shape
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    return image + gauss


a = os.listdir("C:\\Users\\Roy\\PycharmProjects\\ex5-royschossberge\\data_set\\datasets\\image_dataset\\train\\")

for i in range(len(a)):
    a[i] = "C:\\Users\\Roy\\PycharmProjects\\ex5-royschossberge\\data_set\\datasets\\image_dataset\\train\\" + a[i]

gen = load_dataset(a, 10, noisy, (100, 100))
(source_batch, target_batch) = next(gen)

for i in range(10):
    orig = target_batch[i] + 0.5
    dem = source_batch[i] + 0.5

    plt.imshow(orig.reshape((100, 100)), cmap=plt.cm.gray)
    plt.show()

    plt.imshow(dem.reshape((100, 100)), cmap=plt.cm.gray)
    plt.show()
    print(i)

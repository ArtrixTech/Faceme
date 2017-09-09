import numpy,os,base64
from PIL import Image


def mosaic_array(np_arr_1, np_arr_2):
    if isinstance(
            np_arr_1,
            numpy.ndarray) and isinstance(
            np_arr_2,
            numpy.ndarray):
        l1 = list(np_arr_1)
        l2 = list(np_arr_2)
        l1.extend(l2)
        return numpy.array(l1)
    else:
        return numpy.array([])


def img_to_base64(input_img, is_numpy_img_array=False):

    if not is_numpy_img_array:
        with open(input_img, mode="rb") as img:
            b64 = base64.b64encode(input_img.read())
        return b64
    else:
        img = Image.fromarray(input_img)
        img.save("temp_img.jpg")
        with open("temp_img.jpg",mode="rb") as file:
            b64=base64.b64encode(file.read())
        os.remove("temp_img.jpg")
        return b64

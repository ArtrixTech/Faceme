import numpy


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

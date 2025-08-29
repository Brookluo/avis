from typing import Tuple
import numpy as np



def center_crop_image(imdata: np.ndarray, ctr_pix_x, ctr_pix_y, box_len: int):
    """Center crop the image to near the object in a square box.
    The object will not be at center if the image is too close to
    the side. The new object position will be returned

    """
    # x, y = imwcs.world_to_pixel(ctr_coord)
    x = np.rint(ctr_pix_x)
    y = np.rint(ctr_pix_y)
    # Absolutly no paddings, try to move to another direction
    # to patch the information.
    # part = np.zeros((box_len, box_len))
    y_lbound = np.floor(y - box_len / 2).astype(int)
    y_rbound = np.ceil(y + box_len / 2).astype(int)
    x_lbound = np.floor(x - box_len / 2).astype(int)
    x_rbound = np.ceil(x + box_len / 2).astype(int)
    y_lcut = max(y_lbound, 0)
    y_rcut = min(y_rbound, imdata.shape[0])
    x_lcut = max(x_lbound, 0)
    x_rcut = min(x_rbound, imdata.shape[1])
    # print(x, y, y_left, y_right, x_left, x_right)
    # note that left or right padding must be mutually exlcusive
    y_lpad = x_lpad = y_rpad = x_rpad = 0
    new_x = new_y = box_len / 2
    if y_lbound < 0:
        y_lpad = -y_lbound
        y_rcut += y_lpad
        new_y -= y_lpad
    elif y_rbound > imdata.shape[0]:
        y_rpad = y_rbound - imdata.shape[0]
        y_lcut -= y_rpad
        new_y += y_rpad
    if x_lbound < 0:
        x_lpad = -x_lbound
        x_rcut += x_lpad
        new_x -= x_lpad
    elif x_rbound > imdata.shape[1]:
        x_rpad = x_rbound - imdata.shape[1]
        x_lcut -= x_rpad
        new_x += x_rpad
    # print(y_lcut, y_rcut, x_lcut, x_rcut)
    cutout = imdata[y_lcut:y_rcut, x_lcut:x_rcut]
    return cutout, new_x, new_y

import tensorflow as tf
import torch
from einops import rearrange


from groupy.gconv.make_gconv_indices import make_c4_z2_indices, make_c4_p4_indices
from groupy.gconv.tensorflow_gconv.transform_filter import transform_filter_2d_nchw, transform_filter_2d_nhwc

from groupy.gconv.make_gconv_indices import make_c4_z2_indices, make_c4_p4_indices,\
    make_d4_z2_indices, make_d4_p4m_indices, flatten_indices

def transform_filter_3d_nhwc(w, flat_indices, shape_info, validate_indices=True):
    no, nto, ni, nti, n = shape_info
    w_flat = tf.reshape(w, [n * n * n * nti, ni, no])  # shape (n * n * n * nti, ni, no)

    # Do the transformation / indexing operation.
    transformed_w = tf.gather(w_flat, flat_indices,
                              validate_indices=validate_indices)  # shape (nto, nti, n, n, n, ni, no)

    # Put the axes in the right order, and collapse them to get a standard shape filter bank
    transformed_w = tf.transpose(transformed_w, [2, 3, 4, 5, 1, 6, 0])  # shape (n, n, n, ni, nti, no, nto)

    transformed_w = tf.reshape(transformed_w, [n, n, n, ni * nti, no * nto])  # shape (n, n, n, ni * nti, no * nto)

    return transformed_w

def transform_filter_3d(w, flat_indices, shape_info, validate_indices=True):
    no, nto, ni, nti, n = shape_info

    w_flat =

if __name__ == "__main__":
    mapping = {'Z3': 1, 'C4': 4, 'D4': 8, 'O': 24, 'C4H': 8, 'D4H': 16, 'OH': 48}
    nti = mapping['Z3']
    nto = mapping['D4H']
    gconv_indices = make_d4h_z3_indices(ksize=3)
    gconv_indices = flatten_indices_3d(gconv_indices)
    w_shape = (3, 3, 3, 8 * nti, 16)
    gconv_shape_info = (out_channels, nto, in_channels, nti, ksize)

    transform_filter_3d_nhwc()

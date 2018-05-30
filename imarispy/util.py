import numpy as np


def make_thumbnail(array, size=256):
    """ array should be 4D array """
    # TODO: don't just crop to the upper left corner
    mip = np.squeeze(array).max(1)[:3, :size, :size].astype(np.float)
    for i in range(mip.shape[0]):
        mip[i] -= np.min(mip[i])
        mip[i] *= 255 / np.max(mip[i])
    mip = np.pad(mip, ((0, 3 - mip.shape[0]),
                       (0, size - mip.shape[1]),
                       (0, size - mip.shape[2])
                       ), 'constant', constant_values=0)
    mip = np.pad(mip, ((0, 1), (0, 0), (0, 0)), 'constant',
                 constant_values=255).astype('|u1')
    return np.squeeze(mip.T.reshape(1, size, size * 4)).astype('|u1')


def h5str(s, coding='ASCII', dtype='S1'):
    return np.frombuffer(str(s).encode(coding), dtype=dtype)


def subsample_data(data, subsamp):
    return data[0::int(subsamp[0]), 0::int(subsamp[1]), 0::int(subsamp[2])]

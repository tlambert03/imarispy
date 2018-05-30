import h5py
from .imaris import np_to_ims
from .bdv import map_imaris_names_to_bdv, np_to_bdv


def np_to_both(*args, **kwargs):
    hf = h5py.File(np_to_ims(*args, **kwargs))
    map_imaris_names_to_bdv(hf)
    hf.close()
    return

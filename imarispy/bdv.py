import os
import h5py
import numpy as np
from xml.etree import ElementTree as ET
from .util import subsample_data


def np_to_bdv(array,
              fname='myfile',
              subsamp=((1, 1, 1), (1, 2, 2)),
              chunks=((4, 32, 32), (16, 16, 16)),
              compression='gzip'
              ):
    assert len(subsamp) == len(chunks)
    assert all([len(i) == 3 for i in subsamp]), 'Only deal with 3D chunks'
    assert all([len(i) == len(x) for i, x in zip(subsamp, chunks)])
    assert compression in (None, 'gzip', 'lzf', 'szip'), 'Unknown compression type'

    fname = os.path.splitext(fname)[0] + '.h5'

    # force 5D
    # TODO: configure/rearrange axes
    if not array.ndim == 5:
        array = array.reshape(tuple([1] * (5 - array.ndim)) + array.shape)
    nt, nc, nz, ny, nx = array.shape
    nr = len(subsamp)

    with h5py.File(fname, 'a') as hf:

        hf['__DATA_TYPES__/Enum_Boolean'] = np.dtype('bool')
        # hf['__DATA_TYPES__/String_VariableLength'] = h5py.special_dtype(vlen=np.dtype('O'))

        for c in range(nc):
            grp = hf.create_group('s{:02d}'.format(c))
            # resolutions and subdivisions require XYZ axis order
            grp.create_dataset('resolutions', data=np.fliplr(np.array(subsamp)),
                               dtype='<f8',
                               chunks=np.array(subsamp).shape,
                               maxshape=(None, None))
            grp.create_dataset('subdivisions', data=np.fliplr(np.array(chunks)),
                               dtype='<i4',
                               chunks=np.array(chunks).shape,
                               maxshape=(None, None))

        fmt = 't{:05d}/s{:02d}/{}'
        for t in range(nt):
            for c in range(nc):
                data = np.squeeze(array[t, c]).astype(np.uint16)
                for r in range(nr):
                    grp = hf.create_group(fmt.format(t, c, r))
                    subsamp = subsample_data(data, subsamp[r])
                    grp.create_dataset('cells', data=subsamp,
                                       chunks=chunks[r],
                                       maxshape=(None, None, None),
                                       scaleoffset=0,
                                       compression=compression)

    write_bdv_xml(fname, array.shape)
    return


def map_imaris_names_to_bdv(hf):
    """ Takes an Imaris file and creates required links for BDV compatibility"""

    # will be populated with shape (nt, nc, nz, ny, nx) of dataset
    shape = [1, 0, 0, 0, 0]
    KEYS = {
        'x': 4,
        'y': 3,
        'z': 2,
        'numberofchannels': 1,
        'noc': 1,
        'datasettimepoints': 0,
        'filetimepoints': 0,
    }

    def visitor(x, y):
        for name, value in y.attrs.items():
            if name.lower() in KEYS:
                shape[KEYS[name.lower()]] = int(value.tostring().decode('ASCII'))

    hf.visititems(visitor)

    assert all([x > 0 for x in shape[-3:]]), 'Could not detect 3D volume size in HD5 file'
    if shape[1] == 0:
        shape[1] = len(hf['DataSet/ResolutionLevel 0/TimePoint 0'])
    if shape[1] == 0:
        while True:
            if bool(hf.get('DataSetInfo/Channel {}'.format(shape[1]))):
                shape[1] += 1
            else:
                break
    assert shape[1] > 0, 'Could not detect number of channels in HD5 file'

    # detect number of resolution levels
    nr = 0
    while True:
        if bool(hf.get('DataSet/ResolutionLevel {}'.format(nr))):
            nr += 1
        else:
            break

    # detect subsampling and chunking rates
    ress = np.empty((nr, 3))
    subs = np.empty_like(ress)
    for r in range(nr):
        d = hf.get('DataSet/ResolutionLevel {}/TimePoint 0/Channel 0/Data'.format(r))
        ress[r] = np.divide(shape[-3:], d.shape).astype(int)
        subs[r] = d.chunks
        assert d.dtype in (np.uint16, np.int16), 'BDV only supports 16 bit files'

    # write BDV-required datasets
    for c in range(shape[1]):
        grp = hf.require_group('s{:02d}'.format(c))
        # resolutions and subdivisions require XYZ axis order
        grp.require_dataset('resolutions', ress.shape, data=np.fliplr(ress), dtype='<f8',
                            chunks=ress.shape, maxshape=(None, None))
        grp.require_dataset('subdivisions', subs.shape, data=np.fliplr(subs), dtype='<i4',
                            chunks=subs.shape, maxshape=(None, None))

    # perform dataset linking between Imaris and BDV formats
    ims_fmt = '/DataSet/ResolutionLevel {}/TimePoint {}/Channel {}/Data'
    bdv_fmt = 't{:05d}/s{:02d}/{}/cells'
    for t in range(shape[0]):
        for c in range(shape[1]):
            for r in range(nr):
                if not bdv_fmt.format(t, c, r) in hf:
                    hf[bdv_fmt.format(t, c, r)] = hf[ims_fmt.format(r, t, c)]
                    # hf[bdv_fmt.format(t, c, r)] = h5py.SoftLink(ims_fmt.format(r, t, c))

    # create the XML file for BDV
    write_bdv_xml(hf.filename, shape)
    return


def write_bdv_xml(fname, imshape, dx=0.1, dy=0.1, dz=0.25):
    nt, nc, nz, ny, nx = tuple(imshape)
    root = ET.Element('SpimData')
    root.set('version', '0.2')
    bp = ET.SubElement(root, 'BasePath')
    bp.set('type', 'relative')
    bp.text = '.'

    seqdesc = ET.SubElement(root, 'SequenceDescription')
    imgload = ET.SubElement(seqdesc, 'ImageLoader')
    imgload.set('format', 'bdv.hdf5')
    el = ET.SubElement(imgload, 'hdf5')
    el.set('type', 'relative')
    el.text = os.path.basename(fname)
    viewsets = ET.SubElement(seqdesc, 'ViewSetups')
    attrs = ET.SubElement(viewsets, 'Attributes')
    attrs.set('name', 'channel')
    for c in range(nc):
        vs = ET.SubElement(viewsets, 'ViewSetup')
        ET.SubElement(vs, 'id').text = str(c)
        ET.SubElement(vs, 'name').text = 'channel {}'.format(c + 1)
        ET.SubElement(vs, 'size').text = '{} {} {}'.format(nx, ny, nz)
        vox = ET.SubElement(vs, 'voxelSize')
        ET.SubElement(vox, 'unit').text = 'micron'
        ET.SubElement(vox, 'size').text = '{} {} {}'.format(dx, dy, dz)
        a = ET.SubElement(vs, 'attributes')
        ET.SubElement(a, 'channel').text = str(c + 1)
        chan = ET.SubElement(attrs, 'Channel')
        ET.SubElement(chan, 'id').text = str(c + 1)
        ET.SubElement(chan, 'name').text = str(c + 1)
    tpoints = ET.SubElement(seqdesc, 'Timepoints')
    tpoints.set('type', 'range')
    ET.SubElement(tpoints, 'first').text = str(0)
    ET.SubElement(tpoints, 'last').text = str(nt - 1)

    vregs = ET.SubElement(root, 'ViewRegistrations')
    for t in range(nt):
        for c in range(nc):
            vreg = ET.SubElement(vregs, 'ViewRegistration')
            vreg.set('timepoint', str(t))
            vreg.set('setup', str(c))
            vt = ET.SubElement(vreg, 'ViewTransform')
            vt.set('type', 'affine')
            ET.SubElement(vt, 'affine').text = '{} 0.0 0.0 0.0 0.0 {} 0.0 0.0 0.0 0.0 {} 0.0'.format(dx, dy, dz)

    tree = ET.ElementTree(root)
    tree.write(os.path.splitext(fname)[0] + ".xml")

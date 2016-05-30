# -*- coding: utf-8 -*-
"""
Functions to load various properties of interest in analog and spiking neural
networks from disk.

Created on Wed Nov 18 13:38:46 2015

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library
from builtins import open

import os
import sys
import numpy as np
from six.moves import cPickle
from snntoolbox.config import settings, architectures, datasets

standard_library.install_aliases()


def load_weights(filepath):
    """
    Load all layer weights from a HDF5 file.
    """

    import h5py

    f = h5py.File(filepath, mode='r')

    params = []
    for k in f.keys():
        params.append(np.array(f.get(k)))

    f.close()

    return params


def to_categorical(y, nb_classes):
    """
    Convert class vector to binary class matrix.

    If the input ``y`` has shape (``nb_samples``,) and contains integers from 0
    to ``nb_classes``, the output array will be of dimension
    (``nb_samples``, ``nb_classes``).
    """

    y = np.asarray(y, dtype='int32')
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def get_dataset():
    """
    Load a classification dataset.

    Returns
    -------
    dataset: tuple
        The dataset as a tuple containing the training and test sample arrays
        (X_train, Y_train, X_test, Y_test)

    Todo
    ----

    @Iulia: Discuss how to support non-classification datasets.
    """

    import gzip

    assert settings['dataset'] in datasets, \
        "Dataset {} not known. Supported datasets: {}".format(
            settings['dataset'], datasets)

    nb_classes = 10

    if settings['dataset'] == 'mnist':
        fname = settings['dataset'] + '.pkl.gz'
        path = download_dataset(
                fname, origin='https://s3.amazonaws.com/img-datasets/' + fname)

        if path.endswith('.gz'):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')

        if sys.version_info < (3,):
            (X_train, y_train), (X_test, y_test) = cPickle.load(f)
        else:
            (X_train, y_train), (X_test, y_test) = \
                cPickle.load(f, encoding='bytes')

        f.close()

    elif settings['dataset'] == 'cifar10':
        from keras.datasets import cifar10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    elif settings['dataset'] == 'caltech101':
        from snntoolbox.io_utils.datasets import caltech101
        nb_classes = 102

        # Download & untar or get local path
        base_path = caltech101.download(dataset='img-gen-resized')

        # Path to image folder
        base_path = os.path.join(base_path, caltech101.tar_inner_dirname)

        # X_test contains only paths to images
        (X_test, y_test) = caltech101.load_paths_from_files(base_path,
                                                            'X_test.txt',
                                                            'y_test.txt')
        (X_train, y_train), (X_val, y_val) = caltech101.load_cv_split_paths(
                                                                base_path, 0)
        print("Warning: Used only a total of two batch sizes for X_train.")
        X_train = caltech101.load_samples(X_train, 2*settings['batch_size'])
        X_test = caltech101.load_samples(X_test, 2*settings['batch_size'])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)

    return (X_train, Y_train, X_test, Y_test)


def get_reshaped_dataset():
    """
    Load a classification dataset and shape it to fit a specific network model.

    Returns
    -------

    The dataset as a tuple containing the training and test sample arrays
    (X_train, Y_train, X_test, Y_test).
    With data of the form (channels, num_rows, num_cols), ``X_train`` and
    ``X_test`` have dimension (num_samples, channels*num_rows*num_cols)
    for a multi-layer perceptron, and
    (num_samples, channels, num_rows, num_cols) for a convolutional net.
    ``Y_train`` and ``Y_test`` have dimension (num_samples, num_classes).
    """

    (X_train, Y_train, X_test, Y_test) = get_dataset()

    assert settings['architecture'] in architectures, "Network \
        architecture {} not understood. Supported architectures: {}".format(
            settings['architecture'], architectures)
    if settings['architecture'] == 'mlp':
        X_train = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
        X_test = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))
    # Data container has no channel dimension, but we need 4D input for CNN:
    elif settings['architecture'] == 'cnn' and X_train.ndim < 4:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1],
                                  X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1],
                                X_test.shape[2])
    return (X_train, Y_train, X_test, Y_test)


def download_dataset(fname, origin, untar=False):
    """
    Download a dataset, if not already there.

    Parameters
    ----------

    fname: string
        Full filename of dataset, e.g. ``mnist.pkl.gz``.
    origin: string
        Location of dataset, e.g. url
        https://s3.amazonaws.com/img-datasets/mnist.pkl.gz
    untar: boolean, optional
        If ``True``, untar file.

    Returns
    -------

    fpath: string
        The path to the downloaded dataset. If the user has write access to
        ``home``, the dataset will be stored in ``~/.snntoolbox/datasets/``,
        otherwise in ``/tmp/.snntoolbox/datasets/``.

    Todo
    ----

    Test under python2.
    """

    import tarfile
    import shutil
    from six.moves.urllib.error import URLError, HTTPError
    # Under Python 2, 'urlretrieve' relies on FancyURLopener from legacy
    # urllib module, known to have issues with proxy management
    from six.moves.urllib.request import urlretrieve

    datadir_base = os.path.expanduser(os.path.join('~', '.snntoolbox'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.snntoolbox')
    datadir = os.path.join(datadir_base, 'datasets')
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        print("Downloading data from {}".format(origin))
        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise e

    if untar:
        if not os.path.exists(untar_fpath):
            print("Untaring file...\n")
            tfile = tarfile.open(fpath, 'r:gz')
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise e
            tfile.close()
        return untar_fpath

    return fpath

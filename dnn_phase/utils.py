import os
import keras
import keras.backend as K
from keras.layers import Dense, Concatenate, Lambda
from keras.layers import Input, Reshape, Flatten
from keras.layers import BatchNormalization as BN
from keras.models import Model
from keras.initializers import Orthogonal
from keras.utils import Sequence
from keras_tqdm import TQDMCallback
from convsep.dataset import LargeDatasetMulti
import numpy as np
import matplotlib.pyplot as plt
import re

def build_dnn(time_context=11, n_channels=2, freq_bins=2049, 
              n_hiddens=500):
    
    # amplitude
    ax = Input(shape=(n_channels, time_context, freq_bins),
               dtype='float32', name='amplitude')
    r_ax = Reshape((time_context*n_channels, freq_bins))(ax)
    r_ax = Lambda(lambda x: K.permute_dimensions(x,(0,2,1)))(r_ax)
    
    fc_a1 = BN()(Dense(n_hiddens,  activation="relu")(r_ax))    
    fc_a2 = BN()(Dense(n_hiddens, activation="relu")(fc_a1))

    
    # phase
    df_ph = Input(shape=(n_channels, time_context, freq_bins),
               dtype='float32', name='df_phase')
    dt_ph = Input(shape=(n_channels, time_context, freq_bins),
               dtype='float32', name='dt_phase')
    c_df = Concatenate(axis=-2)([df_ph, dt_ph])

    r_df = Reshape((time_context*n_channels*2, freq_bins))(c_df)
    r_df = Lambda(lambda x: K.permute_dimensions(x,(0,2,1)))(r_df)
    
    fc_d1 = BN()(Dense(n_hiddens, activation="relu")(r_df))
    fc_d2 = BN()(Dense(n_hiddens, activation="relu")(fc_d1))
    
    
    # regression
    c_all = Concatenate(axis=-1)([fc_a2, fc_d2])
    
    fc_r1 = BN()(Dense(2, activation="relu")(c_all))
    fc_r1 = Lambda(lambda x: K.permute_dimensions(x,(0,2,1)))(fc_r1)
    
    model = Model(inputs=[ax, df_ph, dt_ph], outputs=fc_r1)
    model.compile(optimizer="adam", loss="mse", metrics=['mae', 'acc'])
    return model


class DataGenerator(Sequence):
    """ Conversion of a LargeDataset class into a Keras Sequence """
    
    def __init__(self, features_path, batch_size=16, time_context=11, 
                 n_channels=2, freq_bins=2049, n_sources=4, 
                 shuffle=True, source_id=0):
        'Initialization'
        self.dataloader = LargeDatasetMulti(
            path_transform_in=features_path, overlap=75,
            nsources=n_sources, batch_size=batch_size,
            batch_memory=batch_size*8, time_context=time_context,
            nprocs=3, mult_factor_in=0.3, mult_factor_out=0.3,
            tensortype='float32', extra_features=True, 
            extra_feat_dim=3, model="p")
        self.dataloader.extra_feat_size = freq_bins
        self.n_points = self.dataloader.total_points
        self.time_context = time_context
        self.batch_size = batch_size
        self.freq_bins = freq_bins
        self.n_channels = n_channels
        self.n_sources = n_sources
        self.source_id = source_id
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_points / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        mag, targets, features = self.dataloader()
        ph, df_ph, dt_ph = features[..., 0], features[..., 1], features[..., 2]
        y = targets[:, self.source_id:self.source_id+self.n_channels, 0, :]
        return {'amplitude': mag, 
                'df_phase': df_ph, 
                'dt_phase': dt_ph}, y

  


    

def plot_history(history):
    if 'acc' in history.history:
        plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.grid(True)

        plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.show()





def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood of size window. New dimensions are added at the end of
    `array` or after the corresponding original dimension.
    
    Parameters
    ----------
    array : array_like
        Array to which the rolling window is applied.
    window : int or tuple
        Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a
        to ignore a dimension in the window.
    asteps : tuple
        Aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows. (Equivalent to slicing result)
    wsteps : int or tuple (same size as window)
        steps for the added window dimensions. These can be 0 to repeat values
        along the axis.
    axes: int or tuple
        If given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2.       
    toend : bool
        If False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array.
    
    Returns
    -------
    A view on `array` which is smaller to fit the windows and has windows added
    dimensions (0s not counting), ie. every point of `array` is an array of size
    window.
    
    Examples
    --------
    >>> a = np.arange(9).reshape(3,3)
    >>> rolling_window(a, (2,2))
    array([[[[0, 1],
             [3, 4]],
            [[1, 2],
             [4, 5]]],
           [[[3, 4],
             [6, 7]],
            [[4, 5],
             [7, 8]]]])
    
    Or to create non-overlapping windows, but only along the first dimension:
    >>> rolling_window(a, (2,0), asteps=(2,1))
    array([[[0, 3],
            [1, 4],
            [2, 5]]])
    
    Note that the 0 is discared, so that the output dimension is 3:
    >>> rolling_window(a, (2,0), asteps=(2,1)).shape
    (1, 3, 2)
    
    This is useful for example to calculate the maximum in all (overlapping)
    2x2 submatrixes:
    >>> rolling_window(a, (2,2)).max((2,3))
    array([[4, 5],
           [7, 8]])
           
    Or delay embedding (3D embedding with delay 2):
    >>> x = np.arange(10)
    >>> rolling_window(x, 3, wsteps=2)
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6],
           [3, 5, 7],
           [4, 6, 8],
           [5, 7, 9]])
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int) # maybe crude to cast to int...
    
    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w
    
    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.") 

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps
        
        if np.any(asteps < 1):
             raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps
    
    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
             raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1 # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape # just renaming...
    
    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window==0] = 1
    
    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape
    
    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps
    
    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _
        
        new_shape = np.zeros(len(shape)*2, dtype=int)
        new_strides = np.zeros(len(shape)*2, dtype=int)
        
        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides
    
    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]
    
    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)



def get_shape(shape_file):
    """
    Reads a .shape file
    """
    with open(shape_file, 'rb') as f:
        line = f.readline().decode('ascii')
        if not line.startswith('#'):
            raise IOError('Failed to find shape in file')
        shape = tuple(map(int, re.findall(r'(\d+)', line)))
        return shape
            


def load_tensor(path, window=None):
    """
    Loads a binary .data file
    """
    if not os.path.isfile(path):
        raise IOError('File does not exist: '+path)
        
    f_in = np.fromfile(path)
    shape = get_shape(path.replace('.data','.shape'))
    f_in = f_in.reshape(shape)
    if window is not None:
        return np.rollaxis(rolling_window(f_in, window=window, toend=False), 1)
    return f_in
        

"""
    This file is part of DeepConvSep.

    Copyright (c) 2014-2017 Marius Miron  <miron.marius at gmail.com>

    DeepConvSep is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DeepConvSep is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the Affero GPL License
    along with DeepConvSep.  If not, see <http://www.gnu.org/licenses/>.
 """

import scipy
import numpy as np
from scipy import io
from scipy.signal import gaussian
from collections import defaultdict
import os
import sys
from os import listdir
from os.path import isfile, join
import itertools
import math
import random
import re
from convsep.util import *
from convsep.transform import transformFFT



class PhaseTransform(transformFFT):
    """
    Preprocessing as describe in

    Improving DNN-based Music Source Separation using Phase Features
    Joachim Muth 1 Stefan Uhlich 2 Nathana¨el Perraudin 3 Thomas Kemp 2 Fabien Cardinaux 2 Yuki Mitsufuji 4
    """

    def __init__(self, ttype='fft', bins=48, frameSize=1024, hopSize=256,
            tffmin=25, tffmax=18000, iscale = 'lin', suffix='',
            sampleRate=44100, window=gaussian, **kwargs):
        super(PhaseTransform, self).__init__(ttype='fft', bins=bins,
                frameSize=frameSize, hopSize=hopSize, tffmin=tffmin,
                tffmax=tffmax, iscale = iscale, suffix=suffix,
                sampleRate=sampleRate, window=window, **kwargs)

    def compute_file(self, audio, phase=True, sampleRate=44100):
        """
        Compute the STFT for a single audio signal

        Parameters
        ----------
        audio : 1D numpy array
            The array comprising the audio signals
        phase : bool, optional
            To return the phase
        sampleRate : int, optional
            The sample rate at which to read the signals
        Yields
        ------
        mag : 3D numpy array
            The features computed for each of the signals in the audio array, e.g. magnitude spectrograms
        phs: 3D numpy array
            The features computed for each of the signals in the audio array, e.g. phase spectrograms
        """
        X = stft_norm(audio, window=self.window, hopsize=float(self.hopSize),
                nfft=float(self.frameSize), fs=float(sampleRate))
        mag = np.abs(X)
        mag = mag  / np.sqrt(self.frameSize) #normalization
        ph = np.angle(X)
        df_ph = np.pad(np.diff(ph, axis=1), ((0,0),(1,0)), 'constant')
        dt_ph = np.pad(np.diff(ph, axis=0), ((1,0),(0,0)), 'constant')
        ph = np.stack([df_ph, dt_ph], axis=2)
        print("Shape", mag.shape, audio.shape, ph.shape, X.shape)
        X = None
        return mag, ph

    def compute_inverse(self, mag, phase, sampleRate=44100):
        """
        Compute the inverse STFT for a given magnitude and phase

        Parameters
        ----------
        mag : 3D numpy array
            The features computed for each of the signals in the audio array, e.g. magnitude spectrograms
        phs: 3D numpy array
            The features computed for each of the signals in the audio array, e.g. phase spectrograms
        sampleRate : int, optional
            The sample rate at which to read the signals
        Yields
        ------
        audio : 1D numpy array
            The array comprising the audio signals
        """
        mag = mag  * np.sqrt(self.frameSize) #normalization
        Xback = mag * np.exp(1j*phase)
        data = istft_norm(Xback, window=self.window, analysisWindow=self.window,
                    hopsize=float(self.hopSize), nfft=float(self.frameSize))
        return data


def stft_norm(data, window=None,
         hopsize=256.0, nfft=2048.0, fs=44100.0):
    """
    X = stft_norm(data,window=sinebell(2048),hopsize=1024.0,
                   nfft=2048.0,fs=44100)

    Computes the short time Fourier transform (STFT) of data.

    Inputs:
        data                  :
            one-dimensional time-series to be analyzed
        window=sinebell(2048) :
            analysis window
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
        fs=44100.0            :
            sampling rate of the signal

    Outputs:
        X                     :
            STFT of data
    """

    # window defines the size of the analysis windows
    lengthWindow = window.size

    lengthData = data.size

    # should be the number of frames by YAAFE:
    numberFrames = int(np.ceil(lengthData / np.double(hopsize)) + 2)
    # to ensure that the data array s big enough,
    # assuming the first frame is centered on first sample:
    newLengthData = int((numberFrames-1) * hopsize + lengthWindow)

    # !!! adding zeros to the beginning of data, such that the first window is
    # centered on the first sample of data
    data = np.concatenate((np.zeros(int(lengthWindow/2.0)), data))

    # zero-padding data such that it holds an exact number of frames
    data = np.concatenate((data, np.zeros(newLengthData - data.size)))

    # the output STFT has nfft/2+1 rows. Note that nfft has to be an even
    # number (and a power of 2 for the fft to be fast)
    numberFrequencies = int(nfft / 2 + 1)

    STFT = np.zeros([numberFrequencies, numberFrames], dtype=complex)

    # storing FT of each frame in STFT:
    for n in np.arange(numberFrames):
        beginFrame = int(n*hopsize)
        endFrame = beginFrame+lengthWindow
        frameToProcess = window*data[beginFrame:endFrame]
        STFT[:,n] = np.fft.rfft(frameToProcess, np.int32(nfft))
        frameToProcess = None

    return STFT.T

def istft_norm(X, window=None,
          analysisWindow=None,
          hopsize=256.0, nfft=2048.0):
    """
    data = istft_norm(X,window=sinebell(2048),hopsize=1024.0,nfft=2048.0,fs=44100)

    Computes an inverse of the short time Fourier transform (STFT),
    here, the overlap-add procedure is implemented.

    Inputs:
        X                     :
            STFT of the signal, to be \"inverted\"
        window=sinebell(2048) :
            synthesis window
            (should be the \"complementary\" window
            for the analysis window)
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)

    Outputs:
        data                  :
            time series corresponding to the given STFT
            the first half-window is removed, complying
            with the STFT computation given in the
            function stft

    """
    X=X.T
    if analysisWindow is None:
        analysisWindow = window

    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = X.shape
    lengthData = int(hopsize*(numberFrames-1) + lengthWindow)

    normalisationSeq = np.zeros(lengthData)

    data = np.zeros(lengthData)

    for n in np.arange(numberFrames):
        beginFrame = int(n * hopsize)
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], np.int32(nfft))
        frameTMP = frameTMP[:lengthWindow]
        normalisationSeq[beginFrame:endFrame] = (
            normalisationSeq[beginFrame:endFrame] +
            window * analysisWindow)
        data[beginFrame:endFrame] = (
            data[beginFrame:endFrame] + window * frameTMP)

    data = data[int(lengthWindow/2.0):]
    normalisationSeq = normalisationSeq[int(lengthWindow/2.0):]
    normalisationSeq[normalisationSeq==0] = 1.

    data = data / normalisationSeq

    return data

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

    You should have received a copy of the GNU General Public License
    along with DeepConvSep.  If not, see <http://www.gnu.org/licenses/>.
 """

import os,sys
from convsep.transform import transformFFT
import convsep.dataset as dataset
from convsep.dataset import LargeDataset, LargeDatasetMulti
import convsep.util as util

from phase_transform import PhaseTransform

import numpy as np
import re
from scipy.signal import blackmanharris, gaussian
import shutil
import time
import six; from six.moves import cPickle
import re
import climate
import six; from six.moves import configparser as ConfigParser

import theano
import theano.tensor as T
import theano.sandbox.rng_mrg
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
from lasagne.layers import ReshapeLayer,Layer
from lasagne.init import Normal
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params


logging = climate.get_logger('trainer')

climate.enable_default_logging()


def load_model(filename):
    with open(filename,'rb') as f:
        params=cPickle.load(f)
    return params

def save_model(filename, model):
    params=lasagne.layers.get_all_param_values(model)
    with open(filename, 'wb') as f:
        cPickle.dump(params,f,protocol=cPickle.HIGHEST_PROTOCOL)
    return None

def build_ca(amp=None, df_ph=None, dt_ph=None, batch_size=16, time_context=11,
        feat_size=2049, nchannels=2, nsources=4):
    """
    Builds a network with lasagne

    Parameters
    ----------
    amp : Theano tensor
        Amplitude input
    phi : Theano tensor
        Phase input
    batch_size : int, optional
        The number of examples in a batch
    time_context : int, optional
        The time context modeled by the network.
    feat_size : int, optional
        The feature size modeled by the network (last dimension of the feature vector)
    Yields
    ------
    l_out : Theano tensor
        The output of the network
    """
#TODO initialisation

    input_shape=(batch_size, nchannels, time_context, feat_size)
    n_hiddens = 500
    out_shape = (batch_size, nchannels*nsources, feat_size)

    # input layer for amplitude features
    a_in = lasagne.layers.InputLayer(shape=input_shape, input_var=amp)

    # scaling
    #FIXME check dim in BiasLayer
    a_bias = lasagne.layers.BiasLayer(a_in)
    a_scale = lasagne.layers.ScaleLayer(a_bias)

    # dense layers
    a_fc_1 = lasagne.layers.DenseLayer(a_scale, n_hiddens,
                                    nonlinearity=lasagne.nonlinearities.rectify)
    a_fc_2 = lasagne.layers.DenseLayer(a_fc_1, n_hiddens,
                                    nonlinearity=lasagne.nonlinearities.rectify)
    #a_out = lasagne.layers.ReshapeLayer(a_fc_2, (out_shape))

    # input layer for phase features
    df_ph_in = lasagne.layers.InputLayer(shape=input_shape, input_var=df_ph)
    dt_ph_in = lasagne.layers.InputLayer(shape=input_shape, input_var=dt_ph)
    phi_in = lasagne.layers.ConcatLayer([df_ph_in, dt_ph_in], axis=1)
    phi_fc_1 = lasagne.layers.DenseLayer(phi_in, n_hiddens,
                                    nonlinearity=lasagne.nonlinearities.rectify)
    phi_fc_2 = lasagne.layers.DenseLayer(phi_fc_1, n_hiddens,
                                    nonlinearity=lasagne.nonlinearities.rectify)

    # concatenation
    concat = lasagne.layers.ConcatLayer([a_fc_2, phi_fc_2], axis=1)

    # output layers
    c_fc = lasagne.layers.DenseLayer(concat, nchannels*nsources*feat_size,
                                    nonlinearity=lasagne.nonlinearities.rectify)
    phi_out = lasagne.layers.ReshapeLayer(c_fc, (out_shape))
    # c_bias = lasagne.layers.BiasLayer(c_fc)
    # c_out = lasagne.layers.NonlinearityLayer(c_bias, nonlinearity=lasagne.nonlinearities.rectify)

    #return phi_out #FIXME
    # DEBUG
    d = lasagne.layers.DenseLayer(a_in, nchannels*nsources*feat_size,
                                    nonlinearity=lasagne.nonlinearities.rectify)
    d_out = lasagne.layers.ReshapeLayer(d, (out_shape))
    return d_out


'''
def train_auto(train, fun, transform, testdir, outdir, num_epochs=30,
                    model="1.pkl", scale_factor=0.3, load=False,
                    skip_train=False, skip_sep=False):
    """
    Trains a network built with \"fun\" with the data generated with \"train\"
    and then separates the files in \"testdir\",writing the result in \"outdir\"

    Parameters
    ----------
    train : Callable, e.g. LargeDataset object
        The callable which generates training data for the network: inputs, target = train()
    fun : lasagne network object, Theano tensor
        The network to be trained
    transform : transformFFT object
        The Transform object which was used to compute the features (see compute_features.py)
    testdir : string, optional
        The directory where the files to be separated are located
    outdir : string, optional
        The directory where to write the separated files
    num_epochs : int, optional
        The number the epochs to train for (one epoch is when all examples in the dataset are seen by the network)
    model : string, optional
        The path where to save the trained model (theano tensor containing the network)
    scale_factor : float, optional
        Scale the magnitude of the files to be separated with this factor
    Yields
    ------
    losser : list
        The losses for each epoch, stored in a list
    """

    logging.info("Building Autoencoder")
    amp = T.tensor4('amp')
    dt_ph = T.tensor4('dt_ph')
    df_ph = T.tensor4('df_ph')
    input_var2 = T.tensor4('inputs')
    target_var2 = T.tensor4('targets')
    rand_num = T.tensor4('rand_num')

    eps=1e-8
    alpha=0.001
    beta=0.01
    beta_voc=0.03

    network2 = fun(amp=amp, df_ph=df_ph, dt_ph=dt_ph, batch_size=train.batch_size,
                    time_context=train.time_context,feat_size=train.input_size)

    if load:
        params=load_model(model)
        lasagne.layers.set_all_param_values(network2,params)

    prediction2 = lasagne.layers.get_output(network2, deterministic=True)

    rand_num = np.random.uniform(size=(train.batch_size,1,train.time_context,train.input_size))

    voc = prediction2[:,0:1,:,:] + eps*rand_num
    bas = prediction2[:,1:2,:,:] + eps*rand_num
    dru = prediction2[:,2:3,:,:] + eps*rand_num
    oth = prediction2[:,3:4,:,:] + eps*rand_num

    mask1=voc/(voc+bas+dru+oth)
    mask2=bas/(voc+bas+dru+oth)
    mask3=dru/(voc+bas+dru+oth)
    mask4=oth/(voc+bas+dru+oth)

    vocals=mask1*input_var2
    bass=mask2*input_var2
    drums=mask3*input_var2
    others=mask4*input_var2

    train_loss_recon_vocals = lasagne.objectives.squared_error(vocals,target_var2[:,0:1,:,:])
    alpha_component = alpha*lasagne.objectives.squared_error(vocals,target_var2[:,1:2,:,:])
    alpha_component += alpha*lasagne.objectives.squared_error(vocals,target_var2[:,2:3,:,:])
    train_loss_recon_neg_voc = beta_voc*lasagne.objectives.squared_error(vocals,target_var2[:,3:4,:,:])

    train_loss_recon_bass = lasagne.objectives.squared_error(bass,target_var2[:,1:2,:,:])
    alpha_component += alpha*lasagne.objectives.squared_error(bass,target_var2[:,0:1,:,:])
    alpha_component += alpha*lasagne.objectives.squared_error(bass,target_var2[:,2:3,:,:])
    train_loss_recon_neg = beta*lasagne.objectives.squared_error(bass,target_var2[:,3:4,:,:])

    train_loss_recon_drums = lasagne.objectives.squared_error(drums,target_var2[:,2:3,:,:])
    alpha_component += alpha*lasagne.objectives.squared_error(drums,target_var2[:,0:1,:,:])
    alpha_component += alpha*lasagne.objectives.squared_error(drums,target_var2[:,1:2,:,:])
    train_loss_recon_neg += beta*lasagne.objectives.squared_error(drums,target_var2[:,3:4,:,:])

    vocals_error=train_loss_recon_vocals.sum()
    drums_error=train_loss_recon_drums.sum()
    bass_error=train_loss_recon_bass.sum()
    negative_error=train_loss_recon_neg.sum()
    negative_error_voc=train_loss_recon_neg_voc.sum()
    alpha_component=alpha_component.sum()

    loss=abs(vocals_error+drums_error+bass_error-negative_error-alpha_component-negative_error_voc)

    params1 = lasagne.layers.get_all_params(network2, trainable=True)

    updates = lasagne.updates.adadelta(loss, params1)

    # val_updates=lasagne.updates.nesterov_momentum(loss1, params1, learning_rate=0.00001, momentum=0.7)

    train_fn = theano.function([amp, df_ph, dt_ph,target_var2], loss, updates=updates,allow_input_downcast=True)

    train_fn1 = theano.function([amp, df_ph, dt_ph,target_var2], [vocals_error,bass_error,drums_error,negative_error,alpha_component,negative_error_voc], allow_input_downcast=True)

    predict_function2=theano.function([amp, df_ph, dt_ph],[vocals,bass,drums,others],allow_input_downcast=True)

    losser=[]
    loss2=[]

    if not skip_train:

        logging.info("Training...")
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            vocals_err=0
            drums_err=0
            bass_err=0
            negative_err=0
            alpha_component=0
            beta_voc=0
            start_time = time.time()
            for batch in range(train.iteration_size):
                mag, target, features = train()
                df_ph, dt_ph = features[..., :]
                jump = inputs.shape[2]
                #mag = np.reshape(mag,(mag.shape[0],2,mag.shape[1],mag.shape[2]))
                #targets=np.ndarray(shape=(mag.shape[0],8,mag.shape[2],inputs.shape[3]))
                #import pdb;pdb.set_trace()
                targets[:,0,:,:]=target[:,:,:jump]
                targets[:,1,:,:]=target[:,:,jump:jump*2]
                targets[:,2,:,:]=target[:,:,jump*2:jump*3]
                targets[:,3,:,:]=target[:,:,jump*3:jump*4]
                target = None

                train_err+=train_fn(mag, df_ph, dt_ph, targets)
                [vocals_erre, bass_erre, drums_erre, negative_erre, alpha,
                    betae_voc] = train_fn1(mag, df_ph, dt_ph, targets)
                vocals_err +=vocals_erre
                bass_err +=bass_erre
                drums_err +=drums_erre
                negative_err +=negative_erre
                beta_voc+=betae_voc
                alpha_component+=alpha
                train_batches += 1

            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err/train_batches))
            losser.append(train_err / train_batches)
            print("  training loss for vocals:\t\t{:.6f}".format(vocals_err/train_batches))
            print("  training loss for bass:\t\t{:.6f}".format(bass_err/train_batches))
            print("  training loss for drums:\t\t{:.6f}".format(drums_err/train_batches))
            print("  Beta component:\t\t{:.6f}".format(negative_err/train_batches))
            print("  Beta component for voice:\t\t{:.6f}".format(beta_voc/train_batches))
            print("  alpha component:\t\t{:.6f}".format(alpha_component/train_batches))
            losser.append(train_err / train_batches)
            save_model(model,network2)

    if not skip_sep:

        logging.info("Separating")
        source = ['vocals','bass','drums','other']
        dev_directory = os.listdir(os.path.join(testdir,"Dev"))
        test_directory = os.listdir(os.path.join(testdir,"Test")) #we do not include the test dir
        dirlist = []
        dirlist.extend(dev_directory)
        dirlist.extend(test_directory)
        for f in sorted(dirlist):
            if not f.startswith('.'):
                if f in dev_directory:
                    song=os.path.join(testdir,"Dev",f,"mixture.wav")
                else:
                    song=os.path.join(testdir,"Test",f,"mixture.wav")
                audioObj, sampleRate, bitrate = util.readAudioScipy(song)

                assert sampleRate == 44100,"Sample rate needs to be 44100"

                audio = (audioObj[:,0] + audioObj[:,1])/2
                audioObj = None
                mag, ph = transform.compute_file(audio,phase=True)

                mag=scale_factor*mag.astype(np.float32)

                batches,nchunks = util.generate_overlapadd(mag,input_size=mag.shape[-1],time_context=train.time_context,overlap=train.overlap,batch_size=train.batch_size,sampleRate=sampleRate)
                output=[]

                batch_no=1
                for batch in batches:
                    batch_no+=1
                    start_time=time.time()
                    output.append(predict_function2(batch))

                output=np.array(output)
                mm=util.overlapadd_multi(output,batches,nchunks,overlap=train.overlap)

                #write audio files
                if f in dev_directory:
                    dirout=os.path.join(outdir,"Dev",f)
                else:
                    dirout=os.path.join(outdir,"Test",f)
                if not os.path.exists(dirout):
                    os.makedirs(dirout)
                for i in range(mm.shape[0]):
                    audio_out=transform.compute_inverse(mm[i,:len(ph)]/scale_factor,ph)
                    if len(audio_out)>len(audio):
                        audio_out=audio_out[:len(audio)]
                    util.writeAudioScipy(os.path.join(dirout,source[i]+'.wav'),audio_out,sampleRate,bitrate)
                    audio_out=None
                audio = None

    return losser
'''

def train_auto(fun,train,transform,testdir,outdir,num_epochs=30,model="1.pkl",scale_factor=0.3,load=False,skip_train=False,skip_sep=False, chunk_size=60,chunk_overlap=2,
    nsamples=40,batch_size=32, batch_memory=50, time_context=30, overlap=25, nprocs=4,mult_factor_in=0.3,mult_factor_out=0.3):
    """
    Trains a network built with \"fun\" with the data generated with \"train\"
    and then separates the files in \"testdir\",writing the result in \"outdir\"

    Parameters
    ----------
    fun : lasagne network object, Theano tensor
        The network to be trained
    transform : PhaseTransform object
        The Transform object which was used to compute the features (see compute_features_DSD100.py)
    testdir : string, optional
        The directory where the files to be separated are located
    outdir : string, optional
        The directory where to write the separated files
    num_epochs : int, optional
        The number the epochs to train for (one epoch is when all examples in the dataset are seen by the network)
    model : string, optional
        The path where to save the trained model (theano tensor containing the network)
    scale_factor : float, optional
        Scale the magnitude of the files to be separated with this factor
    Yields
    ------
    losser : list
        The losses for each epoch, stored in a list
    """

    logging.info("Building Autoencoder")
    amp = T.tensor4('amp')
    dt_ph = T.tensor4('dt_ph')
    df_ph = T.tensor4('df_ph')
    # input_var2 = T.tensor4('inputs')
    input_var = T.tensor4('inputs')
    input_mask = T.tensor4('input_mask')
    target_var = T.tensor4('targets')

    theano_rng = RandomStreams(128)

    eps=1e-12

    sources = ['vocals','bass','drums','other']

    nchannels = int(train.channels_in)
    nsources = int(train.channels_out/train.channels_in)

    print('nchannels: ', nchannels)
    print('nsources: ', nsources)

    input_size = int(float(transform.frameSize) / 2 + 1)

    rand_num = theano_rng.normal(size=(batch_size,nsources,time_context,input_size),
                    avg = 0.0, std = 0.1, dtype=theano.config.floatX)

    network = fun(amp=amp, df_ph=df_ph, dt_ph=dt_ph,
            batch_size=batch_size,time_context=time_context,
            feat_size=input_size, nchannels=nchannels, nsources=nsources)

    if load:
        params=load_model(model)
        lasagne.layers.set_all_param_values(network,params)

    prediction = lasagne.layers.get_output(network, deterministic=True)

    sourceall=[]
    errors_insts = []
    loss = 0

    sep_chann = []

    # prediction example for 2 sources in 2 channels:
    # 0, 1 source 0 in channel 0 and 1
    # 2, 3 source 1 in channel 0 and 1
    ## IDK what is this loop!!!
    for j in range(nchannels):
        #print "j: ", j
        masksum = T.sum(prediction[:,j,:,:],axis=1)
        temp = T.tile(masksum.dimshuffle(0,'x', 1,2),(1,nsources,1,1))
        mask = prediction[:,j,:,:,:] / (temp + eps*rand_num)
        source=mask*T.tile(amp[:,j,:,:],(1,nsources,1,1)) + eps*rand_num
        sourceall.append(source)

        sep_chann.append(source)
        train_loss_recon = lasagne.objectives.squared_error(source,
                                            target_var[:,j,:,:])
        #FIXME only MSE on mag
        errors_inst=abs(train_loss_recon.sum(axis=(0,2,3)))

        errors_insts.append(errors_inst)

        loss=loss+abs(train_loss_recon.sum())

    params1 = lasagne.layers.get_all_params(network, trainable=True)

    updates = lasagne.updates.adadelta(loss, params1)

    train_fn_mse = theano.function([amp,T.abs(target_var)], loss, updates=updates,allow_input_downcast=True)

    train_fn1 = theano.function([amp,T.abs(target_var)], errors_insts, allow_input_downcast=True)

    #----------NEW ILD LOSS CONDITION----------

    rand_num2 = theano_rng.normal( size = (batch_size,nsources,time_context,input_size), avg = 0.0, std = 0.1, dtype=theano.config.floatX) #nsources a primera dim?

    #estimate

    interaural_spec_est = sep_chann[0] / (sep_chann[1] + eps*rand_num2)

    alpha_est = 20*np.log10(abs(interaural_spec_est + eps*rand_num2))
    alpha_est_mean = alpha_est.mean(axis=(0,1,2))

    #groundtruth

    interaural_spec_gt = target_var[:,0::nchannels,:,:] / (target_var[:,1::nchannels,:,:] + eps*rand_num2)

    alpha_gt = 20*np.log10(abs(interaural_spec_gt + eps*rand_num2))
    alpha_gt_mean = alpha_gt.mean(axis=(0,1,2)) #aixo hauria de ser un vector d'una dimensio

    train_loss_ild = lasagne.objectives.squared_error(alpha_est_mean,alpha_gt_mean)

    loss = loss + (abs(train_loss_ild.sum())/500)

    #------------------------------------------

    predict_function=theano.function([amp, df_ph, dt_ph],sourceall,allow_input_downcast=True)

    losser=[]

    if not skip_train:
        logging.info("Training stage 1 (mse)...")
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            errs = np.zeros((nchannels,nsources))
            start_time = time.time()
            for batch in range(train.iteration_size):
                mag, target, features = train()
                df_ph, dt_ph = features[..., :]
                train_err += train_fn_mse(mag, target)
                errs += np.array(train_fn1(mag, df_ph, dt_ph, target))
                train_batches += 1

            logging.info("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            logging.info("  training loss:\t\t{:.6f}".format(train_err/train_batches))
            for j in range(nchannels):
                for i in range(nsources):
                    logging.info("  training loss for "+sources[i]+" in mic "+str(j)+":\t\t{:.6f}".format(errs[j][i]/train_batches))

            model_noILD = model[:-4] + '_noILD' + model[-4:]
            print('model_noILD: ', model_noILD)
            save_model(model_noILD,network)
            losser.append(train_err/train_batches)

#NEW ILD TRAINING---------------------------------------------------------

        params=load_model(model_noILD)
        lasagne.layers.set_all_param_values(network,params)
        params1 = lasagne.layers.get_all_params(network,trainable=True)
        updates = lasagne.updates.adadelta(loss,params1)
        train_fn_ILD = theano.function([amp,target_var],loss,updates=updates,allow_input_downcast=True)

        logging.info("Training stage 2 (ILD)...")

        for epoch in range(int(num_epochs/2)):

            train_err = 0
            train_batches = 0

            start_time = time.time()
            for batch in range(train.iteration_size):
                inputs,target = train()

                train_err+=train_fn_ILD(inputs,target)
                train_batches+=1

            logging.info("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            logging.info("  training loss:\t\t{:.6f}".format(train_err/train_batches))

            save_model(model,network)
            losser.append(train_err/train_batches)

    if not skip_sep:

        logging.info("Separating")

        subsets = ['Dev','Test']
        for sub in subsets:
            for d in sorted(os.listdir(os.path.join(db,'Mixtures',sub))):
                print(os.path.join(os.path.sep,db,'Mixtures',sub,d,'mixture.wav'))
                audio, sampleRate, bitrate = util.readAudioScipy(os.path.join(os.path.sep,db,'Mixtures',sub,d,'mixture.wav'))
                nsamples = audio.shape[0]
                sep_audio = np.zeros((nsamples,len(sources),audio.shape[1]))

                mag,ph=transform.compute_transform(audio,phase=True)
                mag=scale_factor*mag.astype(np.float32)
                #print('mag.shape: ', mag.shape, 'batch_size: ', train.batch_size)
                nframes = mag.shape[-2]

                batches_mag,nchunks = util.generate_overlapadd(mag,input_size=mag.shape[-1],time_context=train.time_context,overlap=train.overlap,batch_size=train.batch_size,sampleRate=sampleRate)
                mag = None

                output=[]
                for b in range(len(batches_mag)):
                    output.append(predict_function(batches_mag[b]))
                output=np.array(output)

                for j in range(audio.shape[1]):
                    mm=util.overlapadd_multi(np.swapaxes(output[:,j:j+1,:,:,:,:],1,3),batches_mag,nchunks,overlap=train.overlap)
                    for i in range(len(sources)):
                        audio_out=transform.compute_inverse(mm[i,:ph.shape[1],:]/scale_factor,ph[j])
                        # if len(sep_audio[:i,j])<len(audio_out):
                        #     print len(sep_audio), len(audio_out), len(audio_out)-len(sep_audio[:i,j])
                        #     sep_audio = np.concatenate(sep_audio,np.zeros(len(audio_out)-len(sep_audio[:i,j])))
                        #     print len(sep_audio), len(audio_out), len(audio_out)-len(sep_audio[:i,j])
                        sep_audio[:,i,j] = audio_out[:len(sep_audio)]

                print('Saving separation: ', outdir)
                if not os.path.exists(os.path.join(outdir)):
                    os.makedirs(os.path.join(outdir))
                    print('Creating model folder')
                if not os.path.exists(os.path.join(outdir,'Sources')):
                    os.makedirs(os.path.join(outdir,'Sources'))
                    print('Creating Sources folder: ', os.path.join(outdir,'Sources'))
                if not os.path.exists(os.path.join(outdir,'Sources',sub)):
                    os.makedirs(os.path.join(outdir,'Sources',sub))
                    print('Creating subset folder')
                if not os.path.exists(os.path.join(outdir,'Sources',sub,d)):
                    os.makedirs(os.path.join(outdir,'Sources',sub,d))
                    print('Creating song folder', os.path.join(outdir,'Sources',sub,d))
                for i in range(len(sources)):
                    print('Final audio file: ', i, os.path.join(outdir,'Sources',sub,d,sources[i]+'.wav'), 'nsamples: ', nsamples,'len sep_audio :', len(sep_audio))
                    util.writeAudioScipy(os.path.join(outdir,'Sources',sub,d,sources[i]+'.wav'),sep_audio[:nsamples,i,:],sampleRate,bitrate)

    return losser


if __name__ == "__main__":
    """
    Given the features computed previusly with compute_features, train a network and perform the separation.

    Parameters
    ----------
    db : string
        The path to the iKala dataset
    nepochs : int, optional
        The number the epochs to train for (one epoch is when all examples in the dataset are seen by the network)
    model : string, optional
        The name of the trained model
    scale_factor : float, optional
        Scale the magnitude of the files to be separated with this factor
    batch_size : int, optional
        The number of examples in a batch (see LargeDataset in dataset.py)
    batch_memory : int, optional
        The number of batches to load in memory at once (see LargeDataset in dataset.py)
    time_context : int, optional
        The time context modeled by the network
    overlap : int, optional
        The number of overlapping frames between adjacent segments (see LargeDataset in dataset.py)
    nprocs : int, optional
        The number of CPU to use when loading the data in parallel: the more, the faster (see LargeDataset in dataset.py)
    """
    if len(sys.argv)>-1:
        climate.add_arg('--db', help="the ikala dataset path")
        climate.add_arg('--model', help="the name of the model to test/save")
        climate.add_arg('--nepochs', help="number of epochs to train the net")
        climate.add_arg('--time_context', help="number of frames for the recurrent/lstm/conv net")
        climate.add_arg('--batch_size', help="batch size for training")
        climate.add_arg('--batch_memory', help="number of big batches to load into memory")
        climate.add_arg('--overlap', help="overlap time context for training")
        climate.add_arg('--nprocs', help="number of processor to parallelize file reading")
        climate.add_arg('--scale_factor', help="scale factor for the data")
        climate.add_arg('--feature_path', help="the path where to load the features from")
        db=None
        kwargs = climate.parse_args()
        if kwargs.__getattribute__('db'):
            db = kwargs.__getattribute__('db')
        else:
            db='/home/marius/Documents/Database/DSD100/'
        if kwargs.__getattribute__('feature_path'):
            feature_path = kwargs.__getattribute__('feature_path')
        else:
            feature_path=os.path.join(db,'transforms','t1')
        assert os.path.isdir(db), "Please input the directory for the DSD100 dataset with --db path_to_DSD100"
        if kwargs.__getattribute__('model'):
            model = kwargs.__getattribute__('model')
        else:
            model="dsd_fft_1024"
        if kwargs.__getattribute__('batch_size'):
            batch_size = int(kwargs.__getattribute__('batch_size'))
        else:
            batch_size = 16
        if kwargs.__getattribute__('batch_memory'):
            batch_memory = int(kwargs.__getattribute__('batch_memory'))
        else:
            batch_memory = 200
        if kwargs.__getattribute__('time_context'):
            time_context = int(kwargs.__getattribute__('time_context'))
        else:
            time_context = 11
        if kwargs.__getattribute__('overlap'):
            overlap = int(kwargs.__getattribute__('overlap'))
        else:
            overlap = 3
        if kwargs.__getattribute__('nprocs'):
            nprocs = int(kwargs.__getattribute__('nprocs'))
        else:
            nprocs = 4
        if kwargs.__getattribute__('nepochs'):
            nepochs = int(kwargs.__getattribute__('nepochs'))
        else:
            nepochs = 40
        if kwargs.__getattribute__('scale_factor'):
            scale_factor = int(kwargs.__getattribute__('scale_factor'))
        else:
            scale_factor = 0.3

    #tt object needs to be the same as the one in compute_features
    # tt = PhaseTransform(frameSize=4096, hopSize=1024, sampleRate=44100, window=gaussian)
    tt = PhaseTransform(frameSize=4096, hopSize=1024, sampleRate=44100, window=gaussian, std=0.4)

    ld1 = LargeDatasetMulti(path_transform_in=feature_path, nsources=4,
                    batch_size=batch_size, batch_memory=batch_memory,
                    time_context=time_context, overlap=overlap, nprocs=nprocs,
                    mult_factor_in=scale_factor, mult_factor_out=scale_factor)
         #sampleRate=tt.sampleRate,tensortype=theano.config.floatX)
    logging.info("  Maximum:\t\t{:.6f}".format(ld1.getMax()))
    logging.info("  Mean:\t\t{:.6f}".format(ld1.getMean()))
    logging.info("  Standard dev:\t\t{:.6f}".format(ld1.getStd()))

    if not os.path.exists(os.path.join(db,'output',model)):
        os.makedirs(os.path.join(db,'output',model))
    if not os.path.exists(os.path.join(db,'models')):
        os.makedirs(os.path.join(db,'models'))

    train_errs = train_auto(train=ld1, fun=build_ca, transform=tt,
                outdir=os.path.join(db,'output',model),
                testdir=os.path.join(db,'Mixtures'),
                model=os.path.join(db,'models',"model_"+model+".pkl"),
                num_epochs=nepochs,
                scale_factor=scale_factor)
    with open(os.path.join(db, "models", "loss_"+model+".data", 'wb')) as f:
        cPickle.dump(train_errs,f,protocol=cPickle.HIGHEST_PROTOCOL)

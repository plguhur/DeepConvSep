import numpy as np
from scipy import io
import os
import sys
from os import listdir
from os.path import isfile, join
import six; from six.moves import cPickle as pickle
import random
import re
import multiprocessing
import convsep.util as util
import climate
import itertools as it

from convsep.dataset import LargeDatasetMulti


logging = climate.get_logger('datasetphase')
climate.enable_default_logging()


class LargeDatasetMultiPhase(LargeDatasetMulti):
    
    def __init__(self, prefix_in="in",prefix_out="out", path_transform_in=None, path_transform_out=None, sampleRate=44100, exclude_list=[],nsamples=0, timbre_model_path=None,
        batch_size=64, batch_memory=1000, time_context=-1, overlap=5, tensortype=float, scratch_path=None, extra_features=False, model="", context=5,pitched=False,save_mask=False,
        log_in=False, log_out=False, mult_factor_in=1., mult_factor_out=1.,nsources=2, pitch_norm=127,nprocs=2,jump=0,
        code_phase="p"):
        self.code_phase = code_phase
        super(LargeDatasetMultiPhase, self).__init__(
            prefix_in=prefix_in,
            prefix_out=prefix_out,
            path_transform_in=path_transform_in,
            path_transform_out=path_transform_out,
            sampleRate=sampleRate,
            exclude_list=exclude_list, nsamples=nsamples, extra_features=extra_features, model=model, context=context,
            batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, tensortype=tensortype, scratch_path=scratch_path, nsources=nsources,jump=jump,
            log_in=log_in, log_out=log_out, mult_factor_in=mult_factor_in, mult_factor_out=mult_factor_out,pitched=pitched,save_mask=save_mask,pitch_norm=pitch_norm,nprocs=nprocs)


    def loadInputOutput(self,id):
        """
        Loads the .data fft file from the hard drive
        """
        allmixinput = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[id]],self.file_list[id]))
        allmixoutput = self.loadTensor(os.path.join(self.path_transform_out[self.dirid[id]],self.file_list[id].replace(self.prefix_in+'_m_',self.prefix_out+'_m_')))
        #FIXME load phases ?
        # allmixoutput = self.loadTensor(os.path.join(self.path_transform_out[self.dirid[id]],self.file_list[id].replace(self.prefix_in+'_m_',self.prefix_out+'_m_')))

        #allmixinput = np.expand_dims(allmixinput[0], axis=0)
        return allmixinput,allmixoutput

    def loadFile(self,id,idxbegin=None,idxend=None):
        """
        reads a .data file and splits into batches
        """
        if self.path_transform_in is not None and self.path_transform_out is not None:
            if idxbegin is None:
                idxbegin = 0
            if idxend is None or idxend==-1:
                idxend = self.num_points[id+1] - self.num_points[id]

            inputs,outputs = self.initOutput(idxend - idxbegin)

            #loads the .data fft file from the hard drive
            allmixinput,allmixoutput = self.loadInputOutput(id)
             #TODO init phase here

            #apply a scaled log10(1+value) function to make sure larger values are eliminated
            if self.log_in==True:
                allmixinput = self.mult_factor_in*np.log10(1.0+allmixinput)
            else:
                allmixinput = self.mult_factor_in*allmixinput
            if self.log_out==True:
                allmixoutput = self.mult_factor_out*np.log10(1.0+allmixoutput)
            else:
                allmixoutput = self.mult_factor_out*allmixoutput
            #TODO should we??

            i = 0
            start = 0

            if self.time_context > allmixinput.shape[1]:
                inputs[0,:,:allmixinput.shape[1],:] = allmixinput
                outputs[0,:,:allmixoutput.shape[1],:] = allmixoutput
                features[0,-1, :] = allfeatures
                if self.pitched:
                    pitches[0, :, :allmixinput.shape[1],:] = self.buildPitch(allmixinput[0],allpitch,start,start+self.time_context)
                if self.save_mask:
                    masks[0, :, :allmixinput.shape[1],:] = self.filterSpec(allmixinput[0],allpitch,start,start+self.time_context)
            else:
                while (start + self.time_context) < allmixinput.shape[1]:
                    if i>=idxbegin and i<idxend:
                        allminput = allmixinput[:,start:start+self.time_context,:] #truncate on time axis so it would match the actual context
                        allmoutput = allmixoutput[:,start:start+self.time_context,:]

                        inputs[i-idxbegin] = allminput
                        outputs[i-idxbegin] = allmoutput

                        if self.extra_features:
                            j=0
                            while (i-j*self.jump-1)>=0 and j<self.context:
                                features[i-idxbegin,self.context-j-1, :] = allfeatures[i-j*self.jump-1,:]
                                j=j+1

                        if self.pitched:
                            pitches[i-idxbegin, :, :allmixinput.shape[1], :] = self.buildPitch(allminput,allpitch,start,start+self.time_context)
                        if self.save_mask:
                            masks[i-idxbegin, :, :allmixinput.shape[1], :] = self.filterSpec(allminput,allpitch,start,start+self.time_context)

                    i = i + 1
                    start = start - self.overlap + self.time_context
                    #clear memory
                    allminput=None
                    allmoutput=None

            #clear memory
            allmixinput=None
            allmixoutput=None
            i=None
            j=None
            start=None
            if self.pitched or self.save_mask:
                allpitch=None
            if self.extra_features:
                allfeatures = None

            result = {'inputs':inputs, 'outputs':outputs, 'pitches':pitches, 'masks':masks, 'features':features}
            inputs = None
            outputs = None
            pitches = None
            masks = None
            features = None
            return result

    def initOutput(self,size):
        """
        Allocate memory for read data, where \"size\" is the number of examples of size \"time_context\"
        """
        inp = np.zeros((size, self.channels_in, self.time_context, self.input_size), dtype=self.tensortype)
        out = np.zeros((size, self.channels_out, self.time_context, self.output_size), dtype=self.tensortype)

        return inp,out

    def initPitches(self,size):
        ptc = np.zeros((size, self.channels_out, self.time_context, self.npitches), dtype=self.tensortype)
        return ptc

    def initMasks(self,size):
        msk = np.zeros((size, self.channels_out, self.time_context, self.input_size), dtype=self.tensortype)
        return msk

    def getFeatureSize(self):
        """
        Returns the feature size of the input and of the output to the neural network
        """
        if self.path_transform_in is not None and self.path_transform_out is not None:
            for i in range(len(self.file_list)):
                if os.path.isfile(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i])):

                    allmix = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i]))
                    self.channels_in = allmix.shape[0]
                    self.input_size=allmix.shape[-1]
                    allmix = None

                    allmixoutput = self.loadTensor(os.path.join(self.path_transform_out[self.dirid[i]],self.file_list[0].replace(self.prefix_in+'_m_',self.prefix_out+'_m_')))
                    self.channels_out = allmixoutput.shape[0]
                    self.output_size = allmixoutput.shape[-1]
                    allmixoutput = None

                    assert self.channels_out % self.channels_in == 0, "number of outputs is not multiple of number of inputs"

                    if self.pitched or self.save_mask:
                        pitch = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i].replace(self.prefix_in+'_m_','_'+self.pitch_code+'_')))
                        if len(pitch.shape)>3:
                            self.nchan = np.minimum(pitch.shape[0],self.channels_in)
                            self.total_inst = int(np.floor(self.channels_out/self.nchan))
                            self.ninst = np.minimum(pitch.shape[1],self.total_inst)#number of pitched instruments (inst for which pitch is defined)
                        else:
                            self.ninst = np.minimum(pitch.shape[0],self.channels_out)
                            self.nchan = 1
                            self.total_inst = self.channels_out
                        self.npitches = 127 #midi notes/pitch granularity
                        pitch=None
                    if self.extra_features:
                        feat = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i].replace(self.prefix_in+'_m_','_'+self.model+'_')))
                        self.extra_feat_size = feat.shape[-1]
                        feat=None



    def updatePath(self, path_in, path_out=None):
        """
        Read the list of .data files in path, compute how many examples we can create from each file, and initialize the output variables
        """
        self.path_transform_in = path_in

        if path_out is None:
            self.path_transform_out = self.path_transform_in
        else:
            self.path_transform_out = path_out

        #we read the file_list from the path_transform_in directory
        self.file_list = [f for k in range(len(self.path_transform_in)) for f in os.listdir(self.path_transform_in[k]) \
            if f.endswith(self.prefix_in+'_m_.data') and os.path.isfile(os.path.join(self.path_transform_out[k],f.replace(self.prefix_in+'_m_',self.prefix_out+'_m_'))) and\
            f.split('_',1)[0] not in self.exclude_list]

        self.dirid = [k for k in range(len(self.path_transform_in)) for f in os.listdir(self.path_transform_in[k]) \
            if f.endswith(self.prefix_in+'_m_.data') and os.path.isfile(os.path.join(self.path_transform_out[k],f.replace(self.prefix_in+'_m_',self.prefix_out+'_m_'))) and\
            f.split('_',1)[0] not in self.exclude_list]

        if self.nsamples>2 and self.nsamples < len(self.file_list):
            ids = np.squeeze(np.random.choice(len(self.file_list), size=self.nsamples, replace=False))
            self.file_list = list([self.file_list[iids] for iids in ids])
            self.dirid = list([self.dirid[iids] for iids in ids])
            ids = None

        self.total_files = len(self.file_list)
        if self.total_files<1:
            raise Exception('Could not find any file in the input directory! Files must end with _m_.data')
        logging.info("found %s files",str(self.total_files))
        self.num_points = np.cumsum(np.array([0]+[self.getNum(i) for i in range(self.total_files)], dtype=int))
        self.total_points = self.num_points[-1]
        self.getFeatureSize()
        self.initBatches()


    def initBatches(self):
        """
        Allocates memory for the output
        """
        self.batch_size = np.minimum(self.batch_size,self.num_points[-1])
        self.iteration_size = int(self.total_points / self.batch_size)
        self.batch_memory = np.minimum(self.batch_memory,self.iteration_size)
        logging.info("iteration size %s",str(self.iteration_size))
        self._index = 0
        self.findex = 0
        self.nindex = 1
        self.idxbegin = 0
        self.idxend = 0
        self.foffset = 0
        self.mini_index = 0
        self.scratch_index = 0
        self.batch_inputs = np.zeros((self.batch_memory*self.batch_size,self.channels_in,self.time_context,self.input_size), dtype=self.tensortype)
        self.batch_outputs = np.zeros((self.batch_memory*self.batch_size,self.channels_out,self.time_context,self.output_size), dtype=self.tensortype)
        if self.pitched:
            self.batch_pitches = np.zeros((self.batch_memory*self.batch_size,self.channels_out,self.time_context,self.npitches), dtype=self.tensortype)

        if self.save_mask:
            self.batch_masks = np.zeros((self.batch_memory*self.batch_size,self.channels_out,self.time_context,self.input_size), dtype=self.tensortype)

        if self.extra_features == True:
            self.batch_features = np.zeros((self.batch_memory*self.batch_size,self.context,self.extra_feat_size), dtype=self.tensortype)
            self.loadBatches()

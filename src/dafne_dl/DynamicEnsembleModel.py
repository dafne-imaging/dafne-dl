#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a deep learning module that can be serialized and deserialized, and dynamically changed.
Functions for the operation of the class are provided as references to top-level functions.
Such top level functions should define all the imports within themselves (i.e. don't put the imports at the top of the file).
"""
#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import copy
import os
from collections import OrderedDict

import torch
import dill

from .interfaces import DeepLearningClass
from io import BytesIO
from .misc import fn_to_source, torch_state_to, torch_apply_fn_to_state_1


def default_ensemble_weights_to_model_function(modelObj, weights):
    from dafne_dl.misc import torch_state_to
    for ii in range(len(modelObj.model)):
        modelObj.model[ii].load_state_dict(torch_state_to(weights[ii], modelObj.device))


def default_ensemble_model_to_weights_function(modelObj):
    from dafne_dl.misc import torch_apply_fn_to_state_1
    model_2_weights=[]
    for ii in range(len(modelObj.model)):
        model_2_weights_=torch_apply_fn_to_state_1(modelObj.model[ii].state_dict(), lambda x: x.clone())
        model_2_weights.append(model_2_weights_)
    return model_2_weights


def default_ensemble_delta_function(lhs, rhs, threshold=None):
    from dafne_dl.interfaces import IncompatibleModelError
    from dafne_dl.misc import torch_state_to
    if lhs.model_id != rhs.model_id: raise IncompatibleModelError
    lhs_weights = lhs.get_weights()
    rhs_weights_pre = rhs.get_weights()
    rhs_weights=[]
    for ii in range(len(rhs.model)):
        rhs_weights_ = torch_state_to(rhs.get_weights(), lhs.device)
        rhs_weights.append(rhs_weights_)
    new_weights=[]
    for ii in range(len(lhs.model)):
        new_weights_ = OrderedDict()
        for key, value in lhs_weights[ii].items():
            delta = lhs_weights[ii][key] - rhs_weights[ii][key]
            if threshold is not None:
                delta[torch.abs(delta) < threshold] = 0
            new_weights_[key] = delta
        new_weights.append(new_weights_)
    outputObj = lhs.get_empty_copy()
    outputObj.set_weights(new_weights)
    outputObj.is_delta = True
    outputObj.timestamp_id = rhs.timestamp_id # set the timestamp of the original model to identify the base
    return outputObj


def default_ensemble_add_weights_function(lhs, rhs):
    import torch
    from dafne_dl.misc import torch_apply_fn_to_state_2, torch_state_to
    from dafne_dl.interfaces import IncompatibleModelError
    if lhs.model_id != rhs.model_id: raise IncompatibleModelError
    lhs_weights = lhs.get_weights()

    rhs_weights_pre = rhs.get_weights()
    rhs_weights=[]
    for ii in range(len(rhs.model)):
        rhs_weights_ = torch_state_to(rhs_weights_pre[ii], lhs.device)
        rhs_weights.append(rhs_weights_)
    new_weights=[]
    for ii in range(len(rhs.model)):
        new_weights_ = torch_apply_fn_to_state_2(lhs_weights[ii], rhs_weights[ii], torch.add)
        new_weights.append(new_weights_)
    outputObj = lhs.get_empty_copy()
    outputObj.set_weights(new_weights)
    return outputObj


def default_ensemble_multiply_function(lhs, rhs:float):
    from dafne_dl.misc import torch_apply_fn_to_state_1
    if not isinstance(rhs, (int, float)):
        raise NotImplementedError('Incompatible types for multiplication (only multiplication by numeric factor is allowed)')
    lhs_weights = lhs.get_weights()
    # # lhs_weights = default_ensemble_model_to_weights_function(lhs)
    # print(f"Type of lhs_weights: {type(lhs_weights)}")
    # if isinstance(lhs_weights, list):
    #     print(f"Length of lhs_weights: {len(lhs_weights)}")
    #     if len(lhs_weights) > 0:
    #         print(f"Type of first element: {type(lhs_weights[0])}")

    new_weights=[]
    for ii in range(len(lhs.model)): #
        new_weights_ = torch_apply_fn_to_state_1(lhs_weights[ii], lambda x: x * rhs)
        new_weights.append(new_weights_)
    outputObj = lhs.get_empty_copy()
    outputObj.set_weights(new_weights)
    return outputObj


def default_ensemble_weight_copy_function(weights_in):
    from dafne_dl.misc import torch_apply_fn_to_state_1
    weights_out = []
    for ii in range(len(weights_in)): #
        weights_out_=torch_apply_fn_to_state_1(weights_in[ii], lambda x: x.clone())
        weights_out.append(weights_out_)
    return weights_out


class DynamicEnsembleModel(DeepLearningClass):

    """
    Class to represent a deep learning model that can be serialized/deserialized
    """
    def __init__(self, model_id,  # a unique ID to avoid mixing different models
                 init_model_function,  # inits the model. Accepts no parameters and returns the models
                 apply_model_function,  # function that applies the model. Has the object, and image
                 weights_to_model_function = default_ensemble_weights_to_model_function,  # put models weights inside the models.
                 model_to_weights_function = default_ensemble_model_to_weights_function,  # get the weights from the models in a pickable format
                 calc_delta_function = default_ensemble_delta_function,  # calculate the weight delta
                 apply_delta_function = default_ensemble_add_weights_function,  # apply a weight delta
                 weight_copy_function = default_ensemble_weight_copy_function,  # create a deep copy of weights
                 factor_multiply_function = default_ensemble_multiply_function,
                 incremental_learn_function = None,  # function to perform an incremental learning step
                 weights = None,  # initial weights
                 timestamp_id = None,
                 is_delta = False,
                 data_dimensionality = 3,
                 metadata = None,
                 **kwargs):
        DeepLearningClass.__init__(self, metadata)
        self.model = None
        self.model_id = model_id
        self.is_delta = is_delta
        self.data_dimensionality = data_dimensionality
        self.type = "DynamicEnsembleModel"

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # list identifying the external functions that need to be saved with source and serialized
        self.function_mappings = [
            'init_model_function',
            'apply_model_function',
            'weights_to_model_function',
            'model_to_weights_function',
            'calc_delta_function',
            'apply_delta_function',
            'weight_copy_function',
            'factor_multiply_function',
            'incremental_learn_function',
        ]

        # the following sets the internal attributes self.fn = fn, with additionally adding the source to the function
        for fn_name in self.function_mappings:
            self.set_internal_fn(fn_name, locals()[fn_name])

        self.init_model() # initializes the model

        if timestamp_id is None:
            self.reset_timestamp()
        else:
            self.timestamp_id = timestamp_id  # unique timestamp id; used to identify model versions during federated learning

        if weights: self.set_weights(weights)

    def set_internal_fn(self, internal_name, obj):
        #print('Setting', internal_name)
        if callable(obj):
            src = fn_to_source(obj)
            if type(src) == str:
                obj.source = src

        setattr(self, internal_name, obj)

    def init_model(self):
        """
        Initializes the internal model

        Returns
        -------
        None.

        """
        self.model=[]
        for ii in range(len(self.init_model_function())): #
            model_= self.init_model_function()[ii].to(self.device)
            self.model.append(model_)
        
    def set_weights(self, weights):
        """
        Loads the weights in the internal model

        Parameters
        ----------
        weights : whatever is accepted by the model_to_weights_function
            Weights to be loaded into the model

        Returns
        -------
        None.

        """
        self.weights_to_model_function(self, weights)
        
    def get_weights(self):
        return self.model_to_weights_function(self)
        
    def apply_delta(self, other):
        return self.apply_delta_function(self, other)
    
    def calc_delta(self, other, threshold=None):
        return self.calc_delta_function(self, other, threshold)
    
    def apply(self, data):
        return self.apply_model_function(self, data)

    def factor_multiply(self, factor: float):
        return self.factor_multiply_function(self, factor)
    
    def incremental_learn(self, trainingData, trainingOutputs, bs=1, minTrainImages=10):
        self.incremental_learn_function(self, trainingData, trainingOutputs, bs, minTrainImages)
        
    def dump(self, file):
        """
        Dumps the current status of the object, including functions and weights
        
        Parameters
        ----------
        file:
            a file descriptor (open in writable mode)

        Returns
        -------
        Nothing

        """
        outputDict = {
            'model_id': self.model_id,
            'weights': [torch_state_to(weights, 'cpu') for weights in self.get_weights()],
            'timestamp_id': self.timestamp_id,
            'is_delta': self.is_delta,
            'data_dimensionality': self.get_data_dimensionality(),
            'type': self.type,
            'metadata': self.metadata
            }

        # add the internal functions to the dictionary
        for fn_name in self.function_mappings:
            outputDict[fn_name] = fn_to_source(getattr(self, fn_name))

        dill.dump(outputDict, file)
        # print(f'output dict {outputDict}')
    
    def dumps(self) -> bytes:
        file = BytesIO()
        self.dump(file)
        return file.getvalue()
    
    def get_empty_copy(self) -> DynamicEnsembleModel:
        """
        Gets an empty copy (i.e. without weights) of the current object

        Returns
        -------
        DynamicEnsembleModel
            Output copy

        """
        new_model = DynamicEnsembleModel(self.model_id, self.init_model_function, self.apply_model_function,
                                   weights=None, timestamp_id=self.timestamp_id, is_delta=self.is_delta,
                                      data_dimensionality=self.get_data_dimensionality(), metadata=copy.deepcopy(self.metadata))
        for fn_name in self.function_mappings:
            new_model.set_internal_fn(fn_name, getattr(self, fn_name))
        return new_model

    def copy(self) -> DynamicEnsembleModel:
        """
        Gets a copy (i.e. with weights) of the current object

        Returns
        -------
        DynamicEnsembleModel
            Output copy

        """
        model_out = self.get_empty_copy()
        model_out.set_weights( self.weight_copy_function(self.get_weights()) )
        return model_out

    @staticmethod
    def Load(file) -> DynamicEnsembleModel:
        """
        Creates an object from a file

        Parameters
        ----------
        file : file descriptor
            A file descriptor.

        Returns
        -------
        DynamicEnsembleModel
            A new instance of a dynamic model

        """
        from .model_loaders import load_model_from_class
        input_dict = dill.load(file)
        return load_model_from_class(input_dict, DynamicEnsembleModel)

    @staticmethod
    def Loads(b: bytes) -> DynamicEnsembleModel:
        """
        Creates an object from a binary dump

        Parameters
        ----------
        file : bytes
            A sequence of bytes

        Returns
        -------
        DynamicEnsembleModel
            A new instance of a dynamic model

        """
        file = BytesIO(b)
        return DynamicEnsembleModel.Load(file)

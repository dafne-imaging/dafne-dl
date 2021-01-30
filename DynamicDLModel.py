#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a deep learning module that can be serialized and deserialized, and dynamically changed.
Functions for the operation of the class are provided as references to top-level functions.
Such top level functions should define all the imports within themselves (i.e. don't put the imports at the top of the file).

@author: francesco
"""
from __future__ import annotations
from .interfaces import IncompatibleModelError, DeepLearningClass
import dill
from io import BytesIO
import numpy as np
import inspect

def fn_to_source(function):
    """
    Given a function, returns it source. If the source cannot be retrieved, return the object itself
    """
    print('Converting fn to source')
    if function is None: return None
    try:
        return inspect.getsource(function)
    except OSError:
        print('Conversion failed!')
        try:
            src = function.source
            print('Returning embedded source')
            return src
        except:
            pass
    print('Returning bytecode')
    return function # the source cannot be retrieved, return the object itself


def source_to_fn(source):
    """
    Given a source, return the (first) defined function. If the source is not a string, return the object itself
    """
    if type(source) is not str:
        print("source to fn: source is not a string")
        return source
    print("source to fn: source is string")
    locs = {}
    globs = {}
    exec(source, globs, locs)
    for k,v in locs.items():
        if callable(v):
            print('source_to_fn. Found function', k)
            v.source = source
            return v
    return None


def default_keras_weights_to_model_function(modelObj: DynamicDLModel, weights):
    modelObj.model.set_weights(weights)


def default_keras_model_to_weights_function(modelObj: DynamicDLModel):
    return modelObj.model.get_weights()


def default_keras_delta_function(lhs: DynamicDLModel, rhs: DynamicDLModel, threshold=None):
    if lhs.model_id != rhs.model_id: raise IncompatibleModelError
    lhs_weights = lhs.get_weights()
    rhs_weights = rhs.get_weights()
    newWeights = []
    for depth in range(len(lhs_weights)):
        delta = lhs_weights[depth] - rhs_weights[depth]
        if threshold is not None:
            delta[np.abs(delta) < threshold] = 0
        newWeights.append(delta)
    outputObj = lhs.get_empty_copy()
    outputObj.set_weights(newWeights)
    return outputObj


def default_keras_apply_delta_function(lhs: DynamicDLModel, rhs: DynamicDLModel):
    if lhs.model_id != rhs.model_id: raise IncompatibleModelError
    lhs_weights = lhs.get_weights()
    rhs_weights = rhs.get_weights()
    newWeights = []
    for depth in range(len(lhs_weights)):
        newWeights.append(lhs_weights[depth] + rhs_weights[depth])
    outputObj = lhs.get_empty_copy()
    outputObj.set_weights(newWeights)
    return outputObj


def default_keras_weight_copy_function(weights_in):
    weights_out = []
    for layer in weights_in:
        weights_out.append(layer.copy())
    return weights_out


class DynamicDLModel(DeepLearningClass):

    """
    Class to represent a deep learning model that can be serialized/deserialized
    """
    def __init__(self, model_id, # a unique ID to avoid mixing different models
                 init_model_function, # inits the model. Accepts no parameters and returns the model
                 apply_model_function, # function that applies the model. Has the object, and image, and a sequence containing resolutions as parameters
                 weights_to_model_function = default_keras_weights_to_model_function, # put model weights inside the model.
                 model_to_weights_function = default_keras_model_to_weights_function, # get the weights from the model in a pickable format
                 calc_delta_function = default_keras_delta_function, # calculate the weight delta
                 apply_delta_function = default_keras_apply_delta_function, # apply a weight delta
                 weight_copy_function = default_keras_weight_copy_function, # create a deep copy of weights
                 incremental_learn_function = None, # function to perform an incremental learning step
                 weights = None, # initial weights
                 timestamp_id = None): 
        self.model = None
        self.model_id = model_id
        self.init_model_function = init_model_function
        src = fn_to_source(init_model_function)
        if type(src) == str:
            self.init_model_function.source = src

        self.weights_to_model_function = weights_to_model_function
        src = fn_to_source(weights_to_model_function)
        if type(src) == str:
            self.weights_to_model_function.source = src

        self.model_to_weights_function = model_to_weights_function
        src = fn_to_source(model_to_weights_function)
        if type(src) == str:
            self.model_to_weights_function.source = src

        self.apply_model_function = apply_model_function
        src = fn_to_source(apply_model_function)
        if type(src) == str:
            self.apply_model_function.source = src

        self.calc_delta_function = calc_delta_function
        src = fn_to_source(calc_delta_function)
        if type(src) == str:
            self.calc_delta_function.source = src

        self.apply_delta_function = apply_delta_function
        src = fn_to_source(apply_delta_function)
        if type(src) == str:
            self.apply_delta_function.source = src

        self.incremental_learn_function = incremental_learn_function
        src = fn_to_source(incremental_learn_function)
        if type(src) == str:
            self.incremental_learn_function.source = src

        self.weight_copy_function = weight_copy_function
        src = fn_to_source(weight_copy_function)
        if type(src) == str:
            self.weight_copy_function.source = src

        self.init_model() # initializes the model
        self.timestamp_id = timestamp_id  # unique timestamp id; used to identify model versions during federated learning
        if weights: self.set_weights(weights)
        
    def init_model(self):
        """
        Initializes the internal model

        Returns
        -------
        None.

        """
        self.model = self.init_model_function()
        
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
    
    def incremental_learn(self, trainingData, trainingOutputs, bs=5, minTrainImages=5):
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
            'init_model_function': fn_to_source(self.init_model_function),
            'apply_model_function': fn_to_source(self.apply_model_function),
            'weights_to_model_function': fn_to_source(self.weights_to_model_function),
            'model_to_weights_function': fn_to_source(self.model_to_weights_function),
            'calc_delta_function': fn_to_source(self.calc_delta_function),
            'apply_delta_function': fn_to_source(self.apply_delta_function),
            'incremental_learn_function': fn_to_source(self.incremental_learn_function),
            'weights': self.get_weights(),
            "timestamp_id": self.timestamp_id
            }
        
        dill.dump(outputDict, file)
    
    def dumps(self) -> bytes:
        file = BytesIO()
        self.dump(file)
        return file.getvalue()
    
    def get_empty_copy(self) -> DynamicDLModel:
        """
        Gets an empty copy (i.e. without weights) of the current object

        Returns
        -------
        DynamicDLModel
            Output copy

        """
        return DynamicDLModel(self.model_id, self.init_model_function, self.apply_model_function,
                              self.weights_to_model_function, self.model_to_weights_function,
                              self.calc_delta_function, self.apply_delta_function, 
                              self.weight_copy_function, self.incremental_learn_function, 
                              None, self.timestamp_id)

    def copy(self) -> DynamicDLModel:
        """
        Gets a copy (i.e. with weights) of the current object

        Returns
        -------
        DynamicDLModel
            Output copy

        """
        model_out = self.get_empty_copy()
        model_out.set_weights( self.weight_copy_function(self.get_weights()) )
        return model_out

    @staticmethod
    def Load(file) -> DynamicDLModel:
        """
        Creates an object from a file

        Parameters
        ----------
        file : file descriptor
            A file descriptor.

        Returns
        -------
        DynamicDLModel
            A new instance of a dynamic model

        """
        
        inputDict = dill.load(file)
        for k,v in inputDict.items():
            if '_function' in k:
                inputDict[k] = source_to_fn(v) # convert the functions from source
        outputObj = DynamicDLModel(**inputDict)
        return outputObj
        
    @staticmethod
    def Loads(b: bytes) -> DynamicDLModel:
        """
        Creates an object from a binary dump

        Parameters
        ----------
        file : bytes
            A sequence of bytes

        Returns
        -------
        DynamicDLModel
            A new instance of a dynamic model

        """
        file = BytesIO(b)
        return DynamicDLModel.Load(file)

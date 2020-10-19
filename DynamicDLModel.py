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

def defaultKerasWeightsToModelFunction(modelObj: DynamicDLModel, weights):
    modelObj.model.set_weights(weights)
    
def defaultKerasModelToWeightsFunction(modelObj: DynamicDLModel):
    return modelObj.model.get_weights()

def defaultKerasDeltaFunction(lhs: DynamicDLModel, rhs: DynamicDLModel):
    if lhs.modelID != rhs.modelID: raise IncompatibleModelError
    lhs_weights = lhs.getWeights()
    rhs_weights = rhs.getWeights()
    newWeights = []
    for depth in range(len(lhs_weights)):
        newWeights.append(lhs_weights[depth] - rhs_weights[depth])
    outputObj = lhs.getEmptyCopy()
    outputObj.setWeights(newWeights)
    return outputObj

def defaultKerasApplyDeltaFunction(lhs: DynamicDLModel, rhs: DynamicDLModel):
    if lhs.modelID != rhs.modelID: raise IncompatibleModelError
    lhs_weights = lhs.getWeights()
    rhs_weights = rhs.getWeights()
    newWeights = []
    for depth in range(len(lhs_weights)):
        newWeights.append(lhs_weights[depth] + rhs_weights[depth])
    outputObj = lhs.getEmptyCopy()
    outputObj.setWeights(newWeights)
    return outputObj

class DynamicDLModel(DeepLearningClass):
    """
    Class to represent a deep learning model that can be serialized/deserialized
    """
    def __init__(self, modelID, # a unique ID to avoid mixing different models
                 initModelFunction, # inits the model. Accepts no parameters and returns the model
                 applyModelFunction, # function that applies the model. Has the object, and image, and a sequence containing resolutions as parameters
                 weightsToModelFunction = defaultKerasWeightsToModelFunction, # put model weights inside the model.
                 modelToWeightsFunction = defaultKerasModelToWeightsFunction, # get the weights from the model in a pickable format
                 calcDeltaFunction = defaultKerasDeltaFunction, # calculate the weight delta
                 applyDeltaFunction = defaultKerasApplyDeltaFunction, # apply a weight delta
                 incrementalLearnFunction = None, # function to perform an incremental learning step
                 weights = None): # initial weights
        self.model = None
        self.modelID = modelID
        self.initModelFunction = initModelFunction
        self.weightsToModelFunction = weightsToModelFunction
        self.modelToWeightsFunction = modelToWeightsFunction
        self.applyModelFunction = applyModelFunction
        self.calcDeltaFunction = calcDeltaFunction
        self.applyDeltaFunction = applyDeltaFunction
        self.incrementalLearnFunction = incrementalLearnFunction
        self.initModel() # initializes the model
        if weights: self.setWeights(weights)
        
    def initModel(self):
        """
        Initializes the internal model

        Returns
        -------
        None.

        """
        self.model = self.initModelFunction()
        
    def setWeights(self, weights):
        """
        Loads the weights in the internal model

        Parameters
        ----------
        weights : whatever is accepted by the modelToWeightsFunction
            Weights to be loaded into the model

        Returns
        -------
        None.

        """
        self.weightsToModelFunction(self, weights)
        
    def getWeights(self):
        return self.modelToWeightsFunction(self)
        
    def applyDelta(self, other):
        return self.applyDelta(self, other)
    
    def calcDelta(self, other):
        return self.calcDelta(self, other)
    
    def apply(self, data):
        return self.applyModelFunction(self, data)
    
    def incrementalLearn(self, trainingData, trainingOutputs):
        self.incrementalLearnFunction(self, trainingData, trainingOutputs)
        
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
            'modelID': self.modelID,
            'initModelFunction': self.initModelFunction,
            'applyModelFunction': self.applyModelFunction,
            'weightsToModelFunction': self.weightsToModelFunction,
            'modelToWeightsFunction': self.modelToWeightsFunction,
            'calcDeltaFunction': self.calcDeltaFunction,
            'applyDeltaFunction': self.applyDeltaFunction,
            'incrementalLearnFunction': self.incrementalLearnFunction,
            'weights': self.getWeights()
            }
        
        dill.dump(outputDict, file)
    
    def dumps(self) -> bytes:
        file = BytesIO()
        self.dump(file)
        return file.getvalue()
    
    def getEmptyCopy(self) -> DynamicDLModel:
        """
        Gets an empty copy (i.e. without weights) of the current object

        Returns
        -------
        DynamicDLModel
            Output copy

        """
        return DynamicDLModel(self.modelID, self.initModelFunction, self.applyModelFunction, self.weightsToModelFunction, self.modelToWeightsFunction, self.calcDeltaFunction, self.applyDeltaFunction, self.incrementalLearnFunction)
    
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
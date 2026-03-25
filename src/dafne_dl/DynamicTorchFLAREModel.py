import torch
import os

from .DynamicTorchModel import (
    DynamicTorchModel,
    default_torch_model_to_weights_function,
    default_torch_delta_function,
    default_torch_weights_to_model_function,
    default_torch_add_weights_function,
    default_torch_weight_copy_function,
    default_torch_multiply_function
)

from .interfaces import DeepLearningClass
import copy

class DynamicTorchFLAREModel(DynamicTorchModel):
    # Override the static Load method to ensure it returns a DynamicTorchFLAREModel instance
    @staticmethod
    def Load(file) -> 'DynamicTorchFLAREModel':
        """
        Creates a DynamicTorchFLAREModel object from a file

        Parameters
        ----------
        file : file descriptor
            A file descriptor.

        Returns
        -------
        DynamicTorchFLAREModel
            A new instance of a dynamic FLARE model
        """
        from .model_loaders import load_model_from_class
        import dill
        input_dict = dill.load(file)
        return load_model_from_class(input_dict, DynamicTorchFLAREModel)
    
    # Override get_empty_copy to return a DynamicTorchFLAREModel instead of DynamicTorchModel
    def get_empty_copy(self):
        """
        Gets an empty copy (i.e. without weights) of the current object, preserving FLARE-specific attributes.

        Returns
        -------
        DynamicTorchFLAREModel
            Output copy
        """
        new_model = DynamicTorchFLAREModel(
            self.model_id, 
            self.init_model_function, 
            self.apply_model_function,
            data_preprocess_function=self.data_preprocess_function if hasattr(self, 'data_preprocess_function') else None,
            data_postprocess_function=self.data_postprocess_function if hasattr(self, 'data_postprocess_function') else None,
            label_preprocess_function=self.label_preprocess_function if hasattr(self, 'label_preprocess_function') else None,
            weights_to_model_function=self.weights_to_model_function,
            model_to_weights_function=self.model_to_weights_function,
            calc_delta_function=self.calc_delta_function,
            apply_delta_function=self.apply_delta_function,
            weight_copy_function=self.weight_copy_function,
            factor_multiply_function=self.factor_multiply_function,
            incremental_learn_function=self.incremental_learn_function if hasattr(self, 'incremental_learn_function') else None,
            weights=None, 
            timestamp_id=self.timestamp_id, 
            is_delta=self.is_delta,
            data_dimensionality=self.get_data_dimensionality(), 
            metadata=copy.deepcopy(self.metadata)
        )
        
        # Copy all internal functions
        for fn_name in self.function_mappings:
            if hasattr(self, fn_name):
                new_model.set_internal_fn(fn_name, getattr(self, fn_name))
        
        return new_model
    def __init__(self, model_id,  # a unique ID to avoid mixing different models
                 init_model_function,  # inits the model. Accepts no parameters and returns the model
                 apply_model_function,  # function that applies the model. Has the object, and image
                 data_preprocess_function = None,  # function that preprocesses the data
                 data_postprocess_function = None,  # function that postprocesses the data
                 label_preprocess_function = None,  # function that preprocesses the labels for training
                 weights_to_model_function = default_torch_weights_to_model_function,  # put model weights inside the model.
                 model_to_weights_function = default_torch_model_to_weights_function,  # get the weights from the model in a pickable format
                 calc_delta_function = default_torch_delta_function,  # calculate the weight delta
                 apply_delta_function = default_torch_add_weights_function,  # apply a weight delta
                 weight_copy_function = default_torch_weight_copy_function,  # create a deep copy of weights
                 factor_multiply_function = default_torch_multiply_function,
                 incremental_learn_function = None,  # function to perform an incremental learning step
                 weights = None,  # initial weights
                 timestamp_id = None,
                 is_delta = False,
                 data_dimensionality = 2,
                 metadata = None,
                 **kwargs):
        DeepLearningClass.__init__(self, metadata)
        self.model = None
        self.model_id = model_id
        self.is_delta = is_delta
        self.data_dimensionality = data_dimensionality
        self.type = "DynamicTorchFLAREModel"

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
            'data_preprocess_function',  # function that preprocesses the data
            'data_postprocess_function',  # function that postprocesses the data
            'label_preprocess_function',  # function that preprocesses the labels for training
            'apply_model_function',
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

    def can_incremental_learn(self) -> bool:
        return getattr(self, 'incremental_learn_function', None) is not None

    def learn(self, train_dataset, validation_dataset, options=None):
        """
        Train the model on the provided datasets.
        
        Parameters
        ----------
        train_dataset : dataset
            The dataset for training
        validation_dataset : dataset
            The dataset for validation
        options : dict, optional
            Training options
            
        Returns
        -------
        None
        """
        # Check if we have an incremental_learn_function attribute (where the learning function is stored)
        if hasattr(self, 'incremental_learn_function'):
            return self.incremental_learn_function(self, train_dataset, validation_dataset, options)
        else:
            raise NotImplementedError("This model does not have a learn function implemented.")
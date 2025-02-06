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

class DynamicTorchFLAREModel(DynamicTorchModel):
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
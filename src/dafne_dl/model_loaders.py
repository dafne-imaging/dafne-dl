import dill
from .misc import source_to_fn

def load_model_from_class(input_dict, model_class):
    # code patches for on-the-fly conversion of old models to new format
    patches = {
        'from dl': 'from dafne_dl',
        'import dl': 'import dafne_dl'
    }

    for k, v in input_dict.items():
        if '_function' in k:
            #print("Converting function", k)
            input_dict[k] = source_to_fn(v, patches)  # convert the functions from source

    # print(inputDict)
    return model_class(**input_dict)


def generic_load_model(file_descriptor_or_dict):
    if isinstance(file_descriptor_or_dict, dict):
        input_dict = file_descriptor_or_dict
    else:
        input_dict = dill.load(file_descriptor_or_dict)
    model_class = input_dict.get('type', 'DynamicDLModel')
    if model_class == 'DynamicDLModel':
        from dafne_dl.DynamicDLModel import DynamicDLModel as ModelClass
    elif model_class == 'DynamicTorchModel':
        from dafne_dl.DynamicTorchModel import DynamicTorchModel as ModelClass
    elif model_class == 'DynamicTorchFLAREModel':
        from dafne_dl.DynamicTorchFLAREModel import DynamicTorchFLAREModel as ModelClass
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    return load_model_from_class(input_dict, ModelClass)

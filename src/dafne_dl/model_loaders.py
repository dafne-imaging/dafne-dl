import dill
import importlib
import re
from .misc import source_to_fn
import flexidep

APP_STRING = 'network.dafne_dl.model_loaders'

def load_model_from_class(input_dict, model_class):
    # code patches for on-the-fly conversion of old models to new format
    patches = {
        'from dl': 'from dafne_dl',
        'import dl': 'import dafne_dl',
        re.escape("""def make_up_layer(input_layer, down_layer, filters, kernel_size, strides, padding, n_conv_layers=2):
        level_up = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, output_padding=(padding, padding), kernel_regularizer=regularizers.l2(reg))(input_layer)"""): \
        """def make_up_layer(input_layer, down_layer, filters, kernel_size, strides, padding, n_conv_layers=2):
        if padding <= 0:
            output_padding = None
        else:
            output_padding = (padding, padding)
        level_up = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, output_padding=output_padding, kernel_regularizer=regularizers.l2(reg))(input_layer)"""
    }

    for k, v in input_dict.items():
        if '_function' in k:
            #print("Converting function", k)
            #input_dict[k] = source_to_fn(v, patches)  # convert the functions from source
            input_dict[k] = source_to_fn(
                v,
                patches,
                classes={
                    model_class.__name__: model_class
                }
            )

    # print(inputDict)
    return model_class(**input_dict)


def generic_load_model(file_descriptor_or_dict, install_deps=True, app_string=APP_STRING, package_manager=flexidep.PackageManagers.pip):
    if isinstance(file_descriptor_or_dict, dict):
        input_dict = file_descriptor_or_dict
    else:
        input_dict = dill.load(file_descriptor_or_dict)
    dependencies = input_dict.get('dependencies', {})
    if not dependencies:
        metadata = input_dict.get('metadata', {})
        dependencies = metadata.get('dependencies', {})
    if install_deps:
        ensure_dependencies(dependencies, app_string, package_manager)
    model_class = input_dict.get('type', 'DynamicDLModel')
    module_name = f"dafne_dl.{model_class}"
    try:
        module = importlib.import_module(module_name)
        ModelClass = getattr(module, model_class)
    except ModuleNotFoundError as e:
        raise ValueError(f"Unknown model module: {module_name}") from e
    except AttributeError as e:
        raise ValueError(f"Module '{module_name}' does not define class '{model_class}'") from e
    return load_model_from_class(input_dict, ModelClass)


def ensure_dependencies(dependencies, app_string=APP_STRING, package_manager=flexidep.PackageManagers.pip, force_reinstall=False):
    dependency_manager = flexidep.DependencyManager(
        config_file=None,
        config_string=None,
        unique_id=app_string,
        interactive_initialization=False,
        use_gui=False,
        install_local=False,
        package_manager=package_manager,
        extra_command_line='',
    )

    for package, alternative_str in dependencies.items():
        print("Processing package", package)
        dependency_manager.process_single_package(package, alternative_str, force_reinstall=force_reinstall)


def get_medicalvolume_orientation_from_metadata(metadata):
    # check if the model has a specific orientation
    model_orientation = metadata.get('orientation', None)

    if isinstance(model_orientation, str):
        # the orientation is a string (Axial/Transversal, Sagittal, Coronal)
        model_orientation = model_orientation.lower()
        if model_orientation.startswith('a') or model_orientation.startswith('t'):
            model_orientation = ('LR', 'AP', 'SI')
        elif model_orientation.startswith('s'):
            model_orientation = ('AP', 'IS', 'LR')
        elif model_orientation.startswith('c'):
            model_orientation = ('LR', 'SI', 'AP')
        else:
            print("Unknown orientation")
            model_orientation = None

    return model_orientation

def ensure_compatible_orientation(image, metadata, inplace=False):
    original_orientation = image.orientation
    model_orientation = get_medicalvolume_orientation_from_metadata(metadata)

    if model_orientation is not None and model_orientation != original_orientation:
        if inplace:
            image.reformat(model_orientation, inplace=True)
            return image
        else:
            return image.reformat(model_orientation, inplace=False)
    else:
        return image

def ensure_compatible_orientation_inplace(image, metadata):
    return ensure_compatible_orientation(image, metadata, inplace=True)
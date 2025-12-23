#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import hashlib
import inspect
from collections import OrderedDict
import re
import torch

import numpy as np

def get_model_detail(model_detail_dict, classification, property, default=None):
    try:
        return model_detail_dict[classification.split(',')[0].strip()][property]
    except KeyError:
        return default



def calculate_file_hash(file_path, cache_results=False, force_rewrite_cache=False):
    if not cache_results:
        return hashlib.sha256(open(file_path, 'rb').read()).hexdigest()
    
    # check if the hash exists on disk
    hash_file = file_path + '.sha256'
    if not force_rewrite_cache:
        try:
            with open(hash_file, 'r') as f:
                output_hash = f.readline().strip()
        except OSError:
            print('Error while reading from hash file')
        else:
            if len(output_hash) == 64:
                #print('Using cached hash')
                return output_hash
            else:
                print('Invalid stored hash')

    # file does not exist, or stored hash is invalid
    output_hash = calculate_file_hash(file_path, False)  # fallback to calculating hash
    try:
        with open(hash_file, 'w') as f:
            f.write(output_hash)
    except OSError:
        print('Error writing hash to file')
    return output_hash


def calc_dice_score(y_true, y_pred):
    """
    Binary f1. Same results as sklearn f1 binary.

    y_true: 1D / 2D / 3D array
    y_pred: 1D / 2D / 3D array
    """
    intersect = np.sum(y_true * y_pred)  # works because all multiplied by 0 gets 0
    denominator = np.sum(y_true) + np.sum(y_pred)  # works because all multiplied by 0 gets 0
    return (2 * intersect) / (denominator + 1e-6)

def calc_dice_score_3D(y_true, y_pred):
    """ 
    Ignore background Dice calculation.
    """
    epsilon=1e-6

    label = np.max(y_true)  # single label
    if label == 0:
        return -1.0  
    
    y_true_bin = (y_true == label).astype(np.float32)
    y_pred_bin = (y_pred == label).astype(np.float32)

    intersection = np.sum(y_true_bin * y_pred_bin)
    sum_union = np.sum(y_true_bin) + np.sum(y_pred_bin)

    return (2 * intersection) / (sum_union + epsilon)


def fn_to_source(function):
    """
    Given a function, returns it source. If the source cannot be retrieved, return the object itself
    """
    #print('Converting fn to source')
    # print("function: ", function)
    if function is None: return None
    try:
        return inspect.getsource(function)
    except OSError:
        #print('Conversion failed!')
        try:
            src = function.source
            #print('Returning embedded source')
            return src
        except:
            pass
    print('Getting source failed - Returning bytecode')
    return function # the source cannot be retrieved, return the object itself


def source_to_fn(source, patches = {},  classes={}):
    """
    Given a source, return the (first) defined function. If the source is not a string, return the object itself
    """
    #print('source to fn')
    # print("source: ", source)
    if not isinstance(source, str):
        return source
    #print("source to fn: source is string")
    for search, replace in patches.items():
        source = re.sub(search, replace, source)

    locs = {}
    globs = {}
    
    for name, cls in classes.items():
        globs[name] = cls
    
    try:
        exec(source, globs, locs)
    except Exception as e:
        # print("source to fn: exec failed", e)
        return source # the string was just a string apparently, not valid code
    for k,v in locs.items():
        if callable(v):
            #print('source_to_fn. Found function', k)
            v.source = source
            return v
    return source


def torch_apply_fn_to_state_1(state1, fn):
    """
    Applies a function to a states, and returns the result as a new state
    """
    new_state = OrderedDict()
    for key, value in state1.items():
        new_state[key] = fn(value)
    return new_state


def torch_apply_fn_to_state_2(state1, state2, fn):
    """
    Applies a function to two states, and returns the result as a new state
    """
    new_state = OrderedDict()
    for key, value in state1.items():
        new_state[key] = fn(value, state2[key])
    return new_state


def torch_state_to(state, device):
    new_state = OrderedDict()
    for key, value in state.items():
        new_state[key] = value.to(device)
    return new_state

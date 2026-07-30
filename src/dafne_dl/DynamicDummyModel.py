#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model class for external pipeline models that manage their own weights.
Only apply_model_function is serialized into the .model file; the pipeline
is responsible for locating its own weights at runtime.
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

import os
from io import BytesIO

import dill
import torch

from .interfaces import DeepLearningClass
from .misc import fn_to_source


class DynamicDummyModel(DeepLearningClass):
    """
    Weightless model wrapper for external pipelines that manage their own weights.
    Federated learning and weight arithmetic are not supported.
    """

    def __init__(self,
                 model_id,
                 apply_model_function=None,
                 incremental_learn_function=None,
                 timestamp_id=None,
                 data_dimensionality=3,
                 metadata=None,
                 **kwargs):
        DeepLearningClass.__init__(self, metadata)
        self.model_id = model_id
        self.model = None
        self.data_dimensionality = data_dimensionality
        self.type = 'DynamicDummyModel'

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.function_mappings = ['apply_model_function', 'incremental_learn_function']
        for fn_name in self.function_mappings:
            self._set_fn(fn_name, locals()[fn_name])

        if timestamp_id is None:
            self.reset_timestamp()
        else:
            self.timestamp_id = timestamp_id

    def _set_fn(self, name, obj):
        if callable(obj):
            src = fn_to_source(obj)
            if isinstance(src, str):
                obj.source = src
        setattr(self, name, obj)

    def init_model(self):
        pass

    def can_incremental_learn(self) -> bool:
        return getattr(self, 'incremental_learn_function', None) is not None

    def apply(self, data: dict) -> dict:
        return self.apply_model_function(self, data)

    def incremental_learn(self, trainingData, trainingOutputs, bs=5, minTrainImages=5):
        if not self.can_incremental_learn():
            raise NotImplementedError("This model does not support incremental learning")
        self.incremental_learn_function(self, trainingData, trainingOutputs, bs, minTrainImages)

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

    def calc_delta(self, baseModel: DeepLearningClass) -> DeepLearningClass:
        raise NotImplementedError("DynamicDummyModel does not support federated learning")

    def apply_delta(self, delta_model: DeepLearningClass) -> DeepLearningClass:
        raise NotImplementedError("DynamicDummyModel does not support federated learning")

    def factor_multiply(self, factor: float):
        raise NotImplementedError("DynamicDummyModel does not support weight arithmetic")

    def dump(self, file):
        output_dict = {
            'model_id': self.model_id,
            'timestamp_id': self.timestamp_id,
            'data_dimensionality': self.get_data_dimensionality(),
            'type': self.type,
            'metadata': self.metadata,
        }
        for fn_name in self.function_mappings:
            output_dict[fn_name] = fn_to_source(getattr(self, fn_name))
        dill.dump(output_dict, file)

    def dumps(self) -> bytes:
        f = BytesIO()
        self.dump(f)
        return f.getvalue()

    @staticmethod
    def Load(file) -> DynamicDummyModel:
        from .model_loaders import load_model_from_class
        input_dict = dill.load(file)
        return load_model_from_class(input_dict, DynamicDummyModel)

    @staticmethod
    def Loads(b: bytes) -> DynamicDummyModel:
        return DynamicDummyModel.Load(BytesIO(b))

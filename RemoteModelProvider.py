#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import requests
import shutil

from .interfaces import ModelProvider
from .DynamicDLModel import DynamicDLModel


MODEL_NAMES_MAP = {
    'Classifier': 'classifier.model',
    'Thigh': 'thigh.model',
    'Leg': 'leg.model'
    }

class RemoteModelProvider(ModelProvider):
    
    def __init__(self, models_path):
        self.models_path = Path(models_path)

        # todo: put this into a config file
        self.url_base = 'http://localhost:5000/'
        self.api_key = "abc123"

    def load_model(self, modelName: str, store_on_disk=True) -> DynamicDLModel:
        """
        Load latest model from remote server

        Args:
            modelName: Classifier | Thigh | Leg

        Returns:
            DynamicDLModel or None
        """
        print(f"Loading model: {modelName}")
        model_name = modelName.lower()

        # Get the name of the latest model
        r = requests.post(self.url_base + "info_model",
                          json={"model_type": model_name,
                                "api_key": self.api_key})
        if r.ok:
            latest_timestamp = r.json()['latest_timestamp']
        else:
            print("ERROR: Request to server failed")
            print(f"status code: {r.status_code}")
            print(f"message: {r.json()['message']}")

        # Receive model
        r = requests.post(self.url_base + "get_model",
                          json={"model_type": model_name,
                                "timestamp": latest_timestamp,
                                "api_key": self.api_key})
        if r.ok:
            return DynamicDLModel.Loads(r.content)
        else:
            print("ERROR: Request to server failed")
            print(f"status code: {r.status_code}")
            print(f"message: {r.json()['message']}")
            return None
    
    def available_models(self) -> str:
        return list(MODEL_NAMES_MAP.keys())

    def upload_model(self, modelName: str, model: DynamicDLModel):
        """
        Upload model to server
        
        Args:
            modelName: classifier | thigh | leg
            model: DynamicDLModel
        """
        print("Uploading model...")
        files = {'model_binary': model.dumps()}
        r = requests.post(self.url_base + "upload_model",
                          files=files,
                          data={"model_type": modelName,
                                "api_key": self.api_key})
        print(f"status code: {r.status_code}")
        print(f"message: {r.json()['message']}")
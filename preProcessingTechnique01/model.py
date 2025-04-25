# model.py - Implement the core TCN architecture
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Activation
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import seaborn as sns
import string
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def residual_block(x, filters, kernel_size, dilation_rate):
    """Implements a residual block for the TCN"""
    residual = x
    
    # First dilated convolution
    x = Conv1D(filters, kernel_size, padding='causal', 
               dilation_rate=dilation_rate)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    # Second dilated convolution
    x = Conv1D(filters, kernel_size, padding='causal', 
               dilation_rate=dilation_rate)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    # If dimensions don't match, use 1x1 conv to resize residual
    if residual.shape[-1] != x.shape[-1]:
        residual = Conv1D(filters, 1)(residual)
    
    # Add skip connection
    return x + residual

def build_tcn_model(input_shape, output_shape, filters=64, kernel_size=3, 
                    dilations=[1, 2, 4, 8, 16, 32]):
    """Builds the TCN model architecture"""
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Stack of residual blocks with increasing dilation
    for dilation in dilations:
        x = residual_block(x, filters, kernel_size, dilation)
    
    # Output layer
    outputs = Dense(output_shape, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model
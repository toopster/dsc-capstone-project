import pandas as pd
import numpy as np
import os
import json
import librosa
import pathlib
from pathlib import Path
from sklearn.model_selection import train_test_split

def get_filestats(src_directory):    
    '''
    Gets the audio sample file statistics based on a target directory and
    loads into a DataFrame
        
        Params:
            src_directory (str): 
            Target directory containing the *.wav files for generating
            stats on
        
        Returns:
            filestats_df (pandas.core.frame.DataFrame):
            Pandas dataframecontaining audio sample statistics          
    '''       
    src_files = Path.cwd() / src_directory
    filedata = []

    for src_file in src_files.glob('**/*.wav'):
        
        if src_file.is_file():
            filedata.append([src_file.parent.parts[-1], 
                             src_file.stem + src_file.suffix, 
                             librosa.get_duration(filename=src_file),
                             librosa.get_samplerate(src_file)])
   
    if src_directory.find('_transformed') != -1:
        columns = ['sample_utterance', 
                   'sample_filename', 
                   'sample_duration', 
                   'sample_samplerate']
    else:
        columns = ['sample_speaker', 
                   'sample_filename', 
                   'sample_duration', 
                   'sample_samplerate']
        
    filestats_df = pd.DataFrame(data=filedata, columns=columns)
    
    return filestats_df

def load_data(data_path, feature):
    '''
    Load the data from the JSON file depending on selected feature
        
        Params:
            data_path (str): Path to json file containing data
            feature (str): Specific feature requested, accepts either 'MFCCs'
            or 'mel_specs'
            
        Returns:
            X (ndarray): Inputs
            y (ndarray): Targets
    ''' 
    with open(data_path, 'r') as file_path:
        data = json.load(file_path)

    X = np.array(data[feature])
    y = np.array(data['labels'])

    print('Datasets loaded...')
    
    return X, y


def create_train_test(data_path, feature, test_size=0.2, val_size=0.2):
    '''
    Splits the data to create training, test and validation datasets
        
        Params:
            data_path (str): Path to json file containing data
            feature (str): Specific feature requested, accepts either 'MFCCs' or 'mel_specs'
            test_size (float): Test size percentage
            val_size (float): Validation size percentage
            
        Returns:
            X_train (ndarray): Inputs for the training dataset
            y_train (ndarray): Targets for the training dataset
            X_val (ndarray): Inputs for the validation dataset
            y_val (ndarray): Targets for the validation dataset
            X_test (ndarray): Inputs for the test dataset
            y_test (ndarray): Targets for the test dataset
    ''' 
    # Load dataset
    X, y = load_data(data_path, feature)

    # Create train, test and validation splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)

    # Increase the dimension of the array for each split
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    return X_train, y_train, X_val, y_val, X_test, y_test
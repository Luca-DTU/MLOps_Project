from tests import _PATH_DATA
import os
from src.data.make_dataset import yelp_dataset
import pytest

"""
The following tests are for the data folder
The data will be tested for the following:
    1. If the data folder exists
    2. If the proccesed data exists for train and test
    4. test that the shape of the train and test are as expected

"""

data_there = pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")

## script for importing the train and test data when the data folder exists and make_dataset.py is in the src folder

def test_path_data():
    """Test if the data folder exists"""
    assert os.path.exists(_PATH_DATA)

## if the data folder does not exist then write that the dvc pull command should be run
if not os.path.exists(_PATH_DATA):
    print("Data folder does not exist. Please run 'dvc pull' to download the data")

raw_there = pytest.mark.skipif(not os.path.exists(_PATH_DATA+"/raw"), reason="Raw data files not found")
processed_there = pytest.mark.skipif(not os.path.exists(_PATH_DATA+"/processed"), reason="Processed data files not found")

## check if processed data exist and if it does import it for other tests
@processed_there
def test_train_processed_exist():
    """Test if the processed/train folder exists"""
    assert os.path.exists(os.path.join(_PATH_DATA+"/processed/train"))
    """Test if the processed/test folder exists"""
    assert os.path.exists(os.path.join(_PATH_DATA+ "/processed/test"))

if os.path.exists(_PATH_DATA+"/processed"):  
    train_p = yelp_dataset(train=True, in_folder=_PATH_DATA + "/raw", out_folder=_PATH_DATA + "/processed")
    test_p = yelp_dataset(train=False, in_folder=_PATH_DATA + "/raw", out_folder=_PATH_DATA + "/processed")

## check if the shape of the train and test data is as expected
@processed_there
def test_processed_data_shapes():
    """Test if the shape of the train data is as expected"""
    assert train_p.data.shape == (1000, 5)
    """Test if the shape of the test data is as expected"""
    assert test_p.data.shape == (1000, 5)

## test the range of data for the processed data
@processed_there
def test_processed_target_range():
    """Test if the range of the train data is as expected"""
    assert min(train_p.data["label"]) == 0
    assert max(train_p.data["label"]) == 4
    """Test if the range of the test data is as expected"""
    assert min(test_p.data["label"]) == 0
    assert max(test_p.data["label"]) == 4

## test the text data for the processed data
@processed_there
def test_processed_text():
    ## test that the number of unique charecters is as expected
    assert len(set("".join(train_p.data['text']))) >= 26
    assert len(set("".join(test_p.data['text']))) >= 26


















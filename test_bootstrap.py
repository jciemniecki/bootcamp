import pytest
import bootstrap as boot
import pandas as pd
import numpy as np

def test_2D_ndarray():
    with pytest.raises(RuntimeError) as excinfo:
        boot.draw_bs_reps(np.array([[1,2,3,4],[5,6,7,8]]))
    excinfo.match('data must be one dimensional.')
    
def test_string_dtype():
    with pytest.raises(RuntimeError) as excinfo:
        boot.draw_bs_reps('np.array[[1,2,3,4],[5]]')
    excinfo.match('data must be of type numpy array.')
    
def test_list_dtype():
    with pytest.raises(RuntimeError) as excinfo:
        boot.draw_bs_reps([1,2,3,4])
    excinfo.match('data must be of type numpy array.')
    
def test_array_NaN():
    with pytest.raises(RuntimeError) as excinfo:
        boot.draw_bs_reps(np.array([1,2,3,np.nan]))
    excinfo.match('data contains NaN values. change or remove these values to proceed.')
    
def test_array_inf():
    with pytest.raises(RuntimeError) as excinfo:
        boot.draw_bs_reps(np.array([1,2,3,np.inf]))
    excinfo.match('data contains inf values. change or remove these values to proceed.')
    
def test_output_size():
    x = boot.draw_bs_reps(np.array([1,2,3,4]), size=6)
    assert x.shape == (6,)
    
def test_diff_func():
    x = boot.draw_bs_reps(np.array([1,2,3,4]), func=np.std, size=2)
    assert x.shape == (2,)
    
def test_no_func():
    x = boot.draw_bs_reps(np.array([1,2,3,4]), func=None, size=2)
    assert x.shape[0] == 8
import numpy as np
import pandas as pd
import bootcamp_utils

def draw_bs_reps(data, var_name, func=np.mean, size=1):
    """
    This function will draw random bootstrap replicates of 
    data and compute a summary statistic on them, if a function 
    is passed.
    
    Parameters:
    ---------------------
    data : ndarray
        Must be 1D.
        
    var_name : str
        Name of variable being bootstrapped
    
    func: function
        Summary statistic function to be computed on the samples.
        Pass None to return ndarray of bootstrap samples
        
    size: int
        Number of samples to generate.
        
    Returns:
    ---------------------
    ndarray of bootstrap summary stat
    OR 2D ndarray of bootstrap samples.
    """
    #------------------------
    # Test data type and dimensions
    
    if not type(data) == np.ndarray: 
        raise RuntimeError('data must be of type numpy array.')
    
    if data.ndim != 1:
        raise RuntimeError('data must be one dimensional.')
            
    #------------------------
    # Test for NaN's in data
    
    if type(data) == np.ndarray:
        if np.isnan(data).any():
            raise RuntimeError('data contains NaN values. change or remove these values to proceed.')
        
    #------------------------
    # Test for inf's in data
    
    if type(data) == np.ndarray:
        if np.isinf(data).any():
            raise RuntimeError('data contains inf values. change or remove these values to proceed.')
            
    #------------------------
    # Bootstrap code!
    
    if func == None:
        bs_samples = np.empty([size,len(data)])
        
        for i in range(size):
            bs_samples[i] = np.random.choice(data, replace=True, size=len(data))
        
        # This one-liner method chain creates a tidy dataframe of the bs samples
        df = pd.DataFrame(data=bs_samples).transpose().melt().rename(columns={'variable':'bs sample', 'value':var_name})
        grouped = df.groupby('bs sample')
        df['ecdf_y grouped by bs sample'] = grouped[var_name].transform(bootcamp_utils.ecdf_y)
        
        return df
    
    else:
        bs_samples = np.empty(size)
        
        for i in range(size):
            bs_samples[i] = func(np.random.choice(data, replace=True, size=len(data)))
    
        return bs_samples
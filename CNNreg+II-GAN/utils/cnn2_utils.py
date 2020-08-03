import numpy as np

def _nanargmin(arr):
    try:
       return np.nanargmin(arr)
    except ValueError:
       return np.nan
   
def frange(start, stop, step):

    num = start
    _list = []
    while num <= stop:
        _list.append(num)
        num = num + step
    
    return _list
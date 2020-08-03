import numpy as np

def _nanargmin(arr): #return nan if there is no result of the np.nanargmin function
    try:
       return np.nanargmin(arr)
    except ValueError:
       return np.nan
   
def frange(start, stop, step): #creates list between start, top and with a defined step (for non integer steps)

    num = start
    _list = []
    while num <= stop:
        _list.append(num)
        num = num + step
    
    return _list
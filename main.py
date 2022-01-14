import numpy as np
import matplotlib.pyplot as plt
## test virtual env
try:
    print(np.array([1,2]))
    print('numpy is installed')
    try:
        a = plt()
        print('you are not in a virtual environment')
    except:
        print('you are in a virtual environment :)')
except:
    print('numpy not yet installed')

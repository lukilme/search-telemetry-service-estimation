import numpy as np

def gen_random_values(size, base):
    random_array = np.random.rand(size, size)
    return random_array + base
#!/usr/bin/python3

import cuda
import math
import numba
from numba import jit
from numba import cuda
import numpy as np
import numexpr as ne
import time

import sys
print("Python version:", sys.version)

cos_array = np.random.rand(100000000)
sin_array = np.random.rand(100000000)

@jit(parallel=True)
def fast_trig():
  for i in range(100000000):
    math.cos(i)
    math.sin(i)

@jit(parallel=True)
def fast_trig_array(v):
  np.cos(v)
  np.sin(v)

# @numba.jit(target='cuda:0', parallel=True)
# def cuda_fast_trig_array(v):
#   dev_v = cuda.device_array_like(v) # send 'v' to gpu (device); stands for 'device v' i assume
#   math.cos(dev_v)
#   math.sin(dev_v)


@cuda.jit('void(int32[:])')
def cuda_fast_trig_array(v):
  dev_v = cuda.device_array_like(v) # send 'v' to gpu (device); stands for 'device v' i assume
  math.cos(dev_v)
  math.sin(dev_v)


# start_time = time.time()
# cos_array = np.cos(cos_array)
# sin_array = np.sin(sin_array)
# print("---%s seconds ---" % (time.time() - start_time))



# calculate 10,000 cosines and sines without just-in-time (jit) compilation
# start_time = time.time()
# for i in range(10000000):
#   math.cos(i)
#   math.sin(i)
# print("---%s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# fast_trig()
# print("---%s seconds ---" % (time.time() - start_time))

# same comparision, but with multi-dimensional numpy arrays:


start_time = time.time()
a = np.arange(100000000)
ne.evaluate('sin(a)')
ne.evaluate('cos(a)')
print("---%s seconds ---" % (time.time() - start_time))

start_time = time.time()
cos_array = np.cos(cos_array)
sin_array = np.sin(sin_array)
print("---%s seconds ---" % (time.time() - start_time))

start_time = time.time()
fast_trig_array(cos_array)
print("---%s seconds ---" % (time.time() - start_time))

start_time = time.time()
cuda_fast_trig_array(cos_array)
print("---%s seconds ---" % (time.time() - start_time))
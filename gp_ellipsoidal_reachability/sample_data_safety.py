
import numpy as np
from environments import InvertedPendulum
n_data = 50

sample_mean = 0
sample_std = np.array([0.1,0.1,.4])
env = InvertedPendulum()

S = np.zeros((n_data,3))
y = np.zeros((n_data,2))

A,B = 
for i in range(n_data):
    s_a = np.random.randn(3)*sample_std + sample_mean
    _,s_new = env.simulate_onestep(s_a[:2],s_a[-1])

    S[i,:] = s_a
    y[i,] = s_new

np.savez("data.npz",S=S,y=y)

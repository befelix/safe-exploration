import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from environments import InvertedPendulum

def plot_sample_set(x_train,z_all,env):
    """ plot the sample set"""
    
    s_train = x_train[:,:env.n_s]
    n_train = np.shape(s_train)[0]
    
    s_expl = z_all[:,:env.n_s]
    n_it = np.shape(s_expl)[0]
    fig, ax = env.plot_safety_bounds(color = "r")
    
    c_spectrum = viridis(np.arange(n_it))
    ##plot initial dataset    
    for i in range(n_train):
        ax = env.plot_state(ax,s_train[i,:env.n_s],color = c_spectrum[0])
    
    ##plot the data gatehred
    for i in range(n_it):
        ax = env.plot_state(ax,s_expl[i,:env.n_s],color = c_spectrum[i])
        
    return fig, ax

results_path = "thesis_results_dynamic_exploration"
a_T_1 = np.load("{}/res_dynamic_exploration_T_1/res_data.npy".format(results_path)).item()
a_T_2 = np.load("{}/res_dynamic_exploration_T_2/res_data.npy".format(results_path)).item()
a_T_3 = np.load("{}/res_dynamic_exploration_T_3/res_data.npy".format(results_path)).item()
a_T_4 = np.load("{}/res_dynamic_exploration_T_4/res_data.npy".format(results_path)).item()
a_T_5 = np.load("{}/res_dynamic_exploration_T_5/res_data.npy".format(results_path)).item()

infgain_1 = np.sum(a_T_1["inf_gain"],axis=1).squeeze()
infgain_2 = np.sum(a_T_2["inf_gain"],axis=1).squeeze()
infgain_3 = np.sum(a_T_3["inf_gain"],axis=1).squeeze()
infgain_4 = np.sum(a_T_4["inf_gain"],axis=1).squeeze()
infgain_5 = np.sum(a_T_5["inf_gain"],axis=1).squeeze()

n_it = len(infgain_1)
#fig, ax = plt.subplots()
plt.plot(np.arange(n_it),infgain_1,label = "T=1",linewidth = 2)
plt.plot(np.arange(n_it),infgain_2,label = "T=2",linewidth = 2)
plt.plot(np.arange(n_it),infgain_3,label = "T=3",linewidth = 2)
plt.plot(np.arange(n_it),infgain_4,label = "T=4",linewidth = 2)
plt.plot(np.arange(n_it),infgain_5,label = "T=5",linewidth = 2)
plt.legend(loc = 2)

ax = plt.gca()
ax.set_xlabel('iteration')
ax.set_ylabel('information gain')

plt.savefig("{}/inf_gain_dynamic.png".format(results_path))

trainset = np.load("{}/random_rollouts_25.npz".format(results_path))
x_train = trainset["S"]

z_T_1 = a_T_1["z_all"]
z_T_2 = a_T_2["z_all"]
z_T_3 = a_T_3["z_all"]
z_T_4 = a_T_4["z_all"]
z_T_5 = a_T_5["z_all"]
env = InvertedPendulum()

plot_sample_set(x_train,z_T_1,env)
plt.savefig("{}/sample_set_T_1.png".format(results_path))
plot_sample_set(x_train,z_T_2,env)
plt.savefig("{}/sample_set_T_2.png".format(results_path))
plot_sample_set(x_train,z_T_3,env)
plt.savefig("{}/sample_set_T_3.png".format(results_path))
plot_sample_set(x_train,z_T_4,env)
plt.savefig("{}/sample_set_T_4.png".format(results_path))
plot_sample_set(x_train,z_T_5,env)
plt.savefig("{}/sample_set_T_5.png".format(results_path))



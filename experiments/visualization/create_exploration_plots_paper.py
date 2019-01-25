import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import viridis
from environments import InvertedPendulum
from utilities_plotting import adapt_figure_size_from_axes,set_figure_params

plot_trainset = False
set_figure_params()


def create_color_bar(n_iterations,bar_label = "Iteration"):
    fig = plt.figure(figsize=(2, 4.5))
    ax1 = fig.add_axes([0.05, 0.05, 0.2, 0.9])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=1, vmax=n_iterations)

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
	                norm=norm,
	                orientation='vertical')
    cb1.set_label(bar_label)

    return fig, ax1

def plot_sample_set(z_all,env,y_label = False, x_train = None):
    """ plot the sample set"""
    
    
    
    s_expl = z_all[:,:env.n_s]
    n_it = np.shape(s_expl)[0]
    fig, ax = env.plot_safety_bounds(color = "r")
    
    c_spectrum = viridis(np.arange(n_it))
    # plot initial dataset    
    if not x_train is None:
	s_train = x_train[:,:env.n_s]
        n_train = np.shape(s_train)[0]
        for i in range(n_train):
            ax = env.plot_state(ax,s_train[i,:env.n_s],color = c_spectrum[0])
    
    # plot the data gatehred
    for i in range(n_it):
        ax = env.plot_state(ax,s_expl[i,:env.n_s],color = c_spectrum[i])
        
    ax.set_xlabel("Angular velocity $\dot{\\theta}$")
    print(y_label)
    if y_label:
	print("??")
	ax.set_ylabel("Angle $\\theta$")
    fig.set_size_inches(3.6,4.5)
    return fig, ax

results_path = "thesis_results_static_exploration" #"thesis_results_dynamic_exploration" #
sub_folder_path = "res_static_mpc_exploration"#"res_dynamic_exploration"# 
colorbar_label = "Iterations" #"Time step"

a_T_1 = np.load("{}/{}_T_1/res_data.npy".format(results_path,sub_folder_path)).item()
a_T_2 = np.load("{}/{}_T_2/res_data.npy".format(results_path,sub_folder_path)).item()
a_T_3 = np.load("{}/{}_T_3/res_data.npy".format(results_path,sub_folder_path)).item()
a_T_4 = np.load("{}/{}_T_4/res_data.npy".format(results_path,sub_folder_path)).item()
a_T_5 = np.load("{}/{}_T_5/res_data.npy".format(results_path,sub_folder_path)).item()

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
ax.set_xlabel('Iteration')
ax.set_ylabel('Information gain')	
fig = plt.gcf()
fig.set_size_inches(14,10)
plt.savefig("{}/inf_gain.png".format(results_path),bbox_inches ='tight')

x_train = None
if plot_trainset:
    trainset = np.load("{}/random_rollouts_25.npz".format(results_path))
    x_train = trainset["S"]

z_T_1 = a_T_1["z_all"]
z_T_2 = a_T_2["z_all"]
z_T_3 = a_T_3["z_all"]
z_T_4 = a_T_4["z_all"]
z_T_5 = a_T_5["z_all"]
env = InvertedPendulum()

_,ax_T_1 = plot_sample_set(z_T_1,env,True)
#plt.savefig("{}/sample_set_T_1.png".format(results_path))
#_,ax_T_2 = plot_sample_set(z_T_2,env)
#plt.savefig("{}/sample_set_T_2.png".format(results_path))
_,ax_T_3 = plot_sample_set(z_T_3,env)
#plt.savefig("{}/sample_set_T_3.png".format(results_path))
#_,ax_T_4 = plot_sample_set(z_T_4,env,True)
#plt.savefig("{}/sample_set_T_4.png".format(results_path))
_,ax_T_5 =plot_sample_set(z_T_5,env)
#plt.savefig("{}/sample_set_T_5.png".format(results_path))

axes = [ax_T_1,ax_T_2,ax_T_3,ax_T_4,ax_T_5]
adapt_figure_size_from_axes(axes)

for i in range(len(axes)):
    ax = axes[i]
    f = ax.get_figure()
    plt.figure(f.number)
    plt.savefig("{}/sample_set_T_{}.png".format(results_path,i+1))

create_color_bar(n_it,colorbar_label)
plt.savefig("{}/color_bar.png".format(results_path),bbox_inches ='tight')

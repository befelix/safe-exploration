import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import viridis
from environments import InvertedPendulum
from utilities_plotting import adapt_figure_size_from_axes,set_figure_params,hide_spines, cm2inches,set_frame_properties
import matplotlib.gridspec as gridspec

def cost_per_episode_scenario(cost_array,safety_failures,shapes, count_safe_only = True, safe_avg = False, norm = 1):
    """ compute average cost per episode and scenario from cost_array

    """

    n_scen, n_ep, n_steps = shapes

    #safety_failures = np.zeros((n_scen,n_ep))
    print(safety_failures)

    res_sum = np.zeros((n_scen,n_ep)) - 1e5 #initialize with -10000
    for i in range(n_scen):
        for j in range(n_ep):
            if not safety_failures[i,j]:
                res_sum[i,j] = np.sum(cost_array[i,j])/norm
                if np.isnan(res_sum[i,j]):
                    print("Something went wrong:")
                    print(cost_array[i,j])

    if safe_avg:
        res_avg_mean = np.empty((n_ep,))
        res_avg_std = np.empty((n_ep,))
        failure_mask = np.array(safety_failures,dtype = bool)
        for j in range(n_ep):
            safe_indices = np.where(np.invert(failure_mask[:,j]))
            print(safe_indices)
            if len(safe_indices) == 0:
                warnings.warn("No successful rollout in episode {}".format(j))
            res_avg_mean[j] =  np.mean(res_sum[safe_indices,j])
            res_avg_std[j] = np.std(res_sum[safe_indices,j])


        return res_avg_mean,res_avg_std

    return res_sum



def _get_failure_ratio(failures,per_episode = False):
    """ Return the failure ratio of all rollouts

    Returns the ratio of failed rollout among all rollouts
    and all episodes

    Parameters
    ----------
    failures: np.ndarray[boolean]
        List of shape (n_ep,n_scenarios) with entries failure_list[i,j] = True if rollout i in scenario j failed
    per_episode: boolean, optional
        Set to True for failure ratio per episode

    Returns
    -------
    failure_ratio: float
        The ratio of failed rollouts and total rollouts


    """
    n_ep, n_scen = np.shape(failures)

    zero_one_failures = np.array(failures,dtype=np.float32)

    if per_episode:
        return np.mean(zero_one_failures,axis = 1)

    return np.mean(zero_one_failures)


plot_trainset = False
set_figure_params()
h_fig_rl_episodes_cm = 3.6
h_fig_bar_cm = 3.6
w_figure_cm = 8.63477
w_full_figure_cm = 17.77747
x_lim_samples = [-1,1]
y_lim_samples = [-1.5,1.5]
rgb_frame = tuple((80/256,80/256,80/256))
h_fig_samples = 4.5

num_fig_samples = 3
h_fig_colorbar = h_fig_samples
w_fig_colorbar = 1.5
w_fig_samples = (w_full_figure_cm - w_fig_colorbar) /num_fig_samples


results_path = "results_journal" #"thesis_results_dynamic_exploration" #
sub_folder_path = "results_rl"#"res_dynamic_exploration"#


colorbar_label = "Iterations" #"Time step"

n_ep = 5
n_steps = 60
n_scen = 10

savedir_safempc = lambda n_safe,n_perf,r,beta_safety: "safempc_CartPole_nsafe={}_nperf={}_r={}_beta_safety={}".format(n_safe,n_perf,r,beta_safety)


a_safe_1_perf_0 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_safempc("1","0","1","2_0"))).item()
a_safe_2_perf_0 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_safempc("2","0","1","2_0"))).item()
a_safe_3_perf_0 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_safempc("3","0","1","2_0"))).item()
a_safe_4_perf_0 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_safempc("4","0","1","2_0"))).item()
#a_safe_5_perf_0 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_safempc("5","0","1","2_0"))).item()

safe_perf_0_list = [a_safe_1_perf_0,a_safe_2_perf_0,a_safe_3_perf_0,a_safe_4_perf_0]


a_safe_1_perf_10 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_safempc("1","15","1","2_0"))).item()
a_safe_2_perf_10 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_safempc("2","15","1","2_0"))).item()
a_safe_3_perf_10 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_safempc("3","15","1","2_0"))).item()
a_safe_4_perf_10 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_safempc("4","15","1","2_0"))).item()
#a_safe_5_perf_10 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_safempc("5","10","1","2_0"))).item()

safe_perf_10_list = [a_safe_1_perf_10,a_safe_2_perf_10,a_safe_3_perf_10,a_safe_4_perf_10]
#safe_perf_10_list = [a_safe_1_perf_10,a_safe_2_perf_10,a_safe_3_perf_10,a_safe_4_perf_10]#,a_safe_5_perf_10]

for k,v in a_safe_1_perf_0.items():
    print(k)
#print(np.array(a_safe_1_perf_10["cc_all"])[0,0])
#print(np.array(a_safe_1_perf_10["safety_failure_all"]))
#print(np.array(a_safe_1_perf_10["X_all"])[1,4])
#print(np.array(a_safe_1_perf_10["X_all"])[0,2][:,2])


# Compute average cost per episode for each n_safe with n_perf=0 and n_perf=10 (two plots)
n_settings_safe = 4
#_, n_ep,_ = np.array(a_safe_1_perf_0["cc_all"]).shape

normalizer = 100
print(a_safe_3_perf_10["safety_failure_all"])

avg_cost_no_perf = np.zeros((n_settings_safe,n_ep))
std_cost_no_perf = np.zeros((n_settings_safe,n_ep))
for i in range(len(safe_perf_0_list)):
    avg_cost_no_perf[i], std_cost_no_perf[i] = cost_per_episode_scenario(np.array(safe_perf_0_list[i]["cc_all"]),np.array(safe_perf_0_list[i]["safety_failure_all"]),(n_scen,n_ep,n_steps),safe_avg = True, norm = normalizer)

avg_cost_10_perf = np.zeros((n_settings_safe,n_ep))
std_cost_10_perf= np.zeros((n_settings_safe,n_ep))
for i in range(len(safe_perf_10_list)):
    print(safe_perf_10_list[i]["safety_failure_all"])
    avg_cost_10_perf[i],std_cost_10_perf[i] = cost_per_episode_scenario(np.array(safe_perf_10_list[i]["cc_all"]),np.array(safe_perf_10_list[i]["safety_failure_all"]),(n_scen,n_ep,n_steps),safe_avg = True, norm = normalizer)

n_it = n_ep


Ts = np.arange(4)+1

fin_cost_perf_list = avg_cost_10_perf[:,-1]
fin_cost_no_perf_list = avg_cost_no_perf[:,-1]

N = 4
ind = np.arange(N)  # the x locations for the groups
width = 0.4       # the width of the bars

fig = plt.figure()
fig.set_size_inches(cm2inches(w_figure_cm),cm2inches(h_fig_bar_cm))
ax = fig.add_subplot(111)

#print(cost_perf_list)
rects1 = ax.bar(ind, fin_cost_no_perf_list, width, color='b',log = False)
rects2 = ax.bar(ind+width, fin_cost_perf_list, width, color='g',log = False)
print(fin_cost_perf_list)
ax.set_ylabel('$C_{ep}$')
ax.set_xticks(ind+width)
ax.set_ylim([0,500])
x_tick_labels = ["T={}".format(i) for i in Ts]
ax.set_xticklabels( x_tick_labels)
#ax.legend( (rects1[0], rects2[0]), ("Standard", "Performance") )
#ax.set_yscale('log')


def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

#set_figure_params()

hide_spines(ax)
set_frame_properties(ax,rgb_frame,0.3)
#plt.tight_layout(h_pad=5)

plt.savefig("bar_plot_rl_std_perf.pdf",bbox_inches ='tight')
#plt.show()


set_figure_params()
fig = plt.figure()

ax = fig.add_subplot(111)


#ax_T_1 = plt.subplot2grid((1, 7), (0, 0), colspan=2)
#ax_T_4 = plt.subplot2grid((1, 7), (0, 2), colspan=2)
#ax_T_5 = plt.subplot2grid((1, 7), (0, 4), colspan=2)
#ax_cb = plt.subplot2grid((1, 7), (0, 6))

y_lim = [100,450]

ax.set_xlabel('Episode')
ax.set_ylabel('$C_{ep}$')
#ax.set_yscale('log')

ax.plot(np.arange(1,n_it+1),avg_cost_10_perf[0],"--o",label = "T=1",linewidth = 1)
ax.plot(np.arange(1,n_it+1),avg_cost_10_perf[1],"--o",label = "T=2",linewidth = 1)
ax.plot(np.arange(1,n_it+1),avg_cost_10_perf[2],"--o",label = "T=3",linewidth = 1)


ax.set_ylim(y_lim)
hide_spines(ax)
ax.set_xticks(np.arange(1,n_it+1))
ax.set_yticks([100,200,300,400])

fig.set_size_inches(cm2inches(w_figure_cm),cm2inches(h_fig_rl_episodes_cm))
plt.savefig("plot_rl_perf_episodes.pdf",bbox_inches ='tight')


H_cautious_mpc = [8,10,12]
savedir_cautious_mpc = lambda beta_safety,n_perf: "cautious_mpc_CartPole_T={}_beta_safety={}".format(beta_safety,n_perf)
caut_mpc_H_5 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_cautious_mpc("5","2_0"))).item()
caut_mpc_H_8 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_cautious_mpc("8","2_0"))).item()
caut_mpc_H_10 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_cautious_mpc("10","2_0"))).item()
caut_mpc_H_12 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_cautious_mpc("12","2_0"))).item()
caut_mpc_H_15 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_cautious_mpc("15","2_0"))).item()
#caut_mpc_H_20 = np.load("{}/{}/{}/results_episode.npy".format(results_path,sub_folder_path,savedir_cautious_mpc("20","2_0"))).item()


print("--- Results cautious MPC --")
print("H = 5:")
print(_get_failure_ratio(caut_mpc_H_5["safety_failure_all"]))
avg_cost_H5, _ = cost_per_episode_scenario(np.array(caut_mpc_H_5["cc_all"]),np.array(caut_mpc_H_5["safety_failure_all"]),(n_scen,n_ep,n_steps),safe_avg = True, norm = normalizer)
print(avg_cost_H5)
print("H = 8:")
print(_get_failure_ratio(caut_mpc_H_8["safety_failure_all"]))
avg_cost_H8, _ = cost_per_episode_scenario(np.array(caut_mpc_H_8["cc_all"]),np.array(caut_mpc_H_8["safety_failure_all"]),(n_scen,n_ep,n_steps),safe_avg = True, norm = normalizer)
print(avg_cost_H8)
print("H=10")
print(_get_failure_ratio(caut_mpc_H_10["safety_failure_all"]))
avg_cost_H10, _ = cost_per_episode_scenario(np.array(caut_mpc_H_10["cc_all"]),np.array(caut_mpc_H_10["safety_failure_all"]),(n_scen,n_ep,n_steps),safe_avg = True, norm = normalizer)
print(avg_cost_H10)
print("H=12")

print(_get_failure_ratio(caut_mpc_H_12["safety_failure_all"]))
avg_cost_H12, _ = cost_per_episode_scenario(np.array(caut_mpc_H_12["cc_all"]),np.array(caut_mpc_H_12["safety_failure_all"]),(n_scen,n_ep,n_steps),safe_avg = True, norm = normalizer)
print(avg_cost_H12)
#print(caut_mpc_H_12["cc_all"])
print("H = 15:")
print(_get_failure_ratio(caut_mpc_H_15["safety_failure_all"]))
avg_cost_H15, _ = cost_per_episode_scenario(np.array(caut_mpc_H_15["cc_all"]),np.array(caut_mpc_H_15["safety_failure_all"]),(n_scen,n_ep,n_steps),safe_avg = True, norm = normalizer)
print(avg_cost_H15)

"""
print("H = 20:")
print(_get_failure_ratio(caut_mpc_H_20["safety_failure_all"]))
avg_cost_H20, _ = cost_per_episode_scenario(np.array(caut_mpc_H_20["cc_all"]),np.array(caut_mpc_H_20["safety_failure_all"]),(n_scen,n_ep,n_steps),safe_avg = True, norm = normalizer)
print(avg_cost_H20)
"""
caut_mpc_H_8["safety_failure_all"]
caut_mpc_H_10["safety_failure_all"]
caut_mpc_H_12["safety_failure_all"]





import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("ggplot")

ref_xhats_f = open("ref_xhats.txt", "r")
bpf_xhats_f = open("bpf_xhats.txt", "r")
ml_parameters = open("ml_parameters.txt", "r")
N1s_data = open("N1s_data.txt", "r")
alloc_counters_f = open("alloc_counters.txt", "r")
ref_stds_f = open("ref_stds.txt", "r")

N_data, N_trials, N_ALLOCS, N_MESHES, N_bpf = list(map(int, ml_parameters.readline().split()))



# -------------------------------------------------- BPF results ---------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
ref_xhats = []
for n_data in range(N_data):
	ref_xhats.append(list(map(float, ref_xhats_f.readline().split())))
length = len(ref_xhats[0])
ref_xhats = np.array(ref_xhats).reshape((N_data, length))
bpf_xhats = np.empty((N_data * N_trials, length))
bpf_rmse = np.zeros(N_data * N_trials)
var_bpf_rmse = np.zeros((N_data * N_trials, length))

custm_length = length
# custm_length = 15
for n_trial in range(N_data * N_trials):
	bpf_xhats[n_trial] = list(map(float, bpf_xhats_f.readline().split()))
if custm_length > length:
	raise Exception("Custom length mustn't exceed full length.")

# Take the RMSE at each trial over the custom length data set
data_set = 0
for n_trial in range(N_data * N_trials):
	rmse = np.sqrt(np.mean((ref_xhats[data_set, :custm_length] - bpf_xhats[n_trial, :custm_length]) ** 2))
	bpf_rmse[n_trial] = rmse
	for n in range(length):
		rmse = np.sqrt(np.mean((ref_xhats[data_set, :n + 1] - bpf_xhats[n_trial, :n + 1]) ** 2))
		var_bpf_rmse[n_trial, n] = rmse
	if (n_trial + 1) % N_trials == 0:
		data_set += 1

# Take the mean of the RMSE over N_data * N_trials
mean_bpf_rmse = np.mean(np.log10(bpf_rmse))
mean_bpf_rmse = np.mean(np.log10(var_bpf_rmse), axis=0)

# Read in the stepwise reference stds
ref_stds = np.array(list(map(float, ref_stds_f.readline().split()))).reshape((N_data, length))
ref_stds = np.mean(ref_stds, axis=0)
mean_ref_xhats = np.mean(ref_xhats, axis=0)
N1_0_mean_ml_xhats = np.zeros((N_MESHES, length))



# ------------------------------------------- Multilevel parameters ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
level0s = list(map(int, ml_parameters.readline().split()))
N1s = list(map(int, N1s_data.readline().split()))
alloc_counters = np.array(list(map(int, alloc_counters_f.readline().split())))
max_allocs = np.max(alloc_counters)
ml_xhats = np.zeros((N_data * N_trials, length))
ml_rmse = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials))
var_ml_rmse = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials, length))
mean_ml_rmse = np.zeros((N_MESHES, N_ALLOCS))
var_mean_ml_rmse = np.zeros((N_MESHES, N_ALLOCS, length))
min_mean_ml_rmse = np.empty(length)
max_mean_ml_rmse = np.empty(length)

# Compute the MLBPF RMSE over all iterates #
for i_mesh in range(N_MESHES):
	nx0 = level0s[i_mesh]
	for n_alloc in range(alloc_counters[i_mesh]):
		N1 = N1s[n_alloc]
		for n_data in range(N_data):
			raw_ml_xhats_f = open("raw_ml_xhats_nx0={}_N1={}_n_data={}.txt".format(nx0, N1, n_data), "r")
			for n_trial in range(N_trials):
				ml_xhats[n_data * N_trials + n_trial] = list(map(float, raw_ml_xhats_f.readline().split()))
		if N1 == 0:
			N1_0_mean_ml_xhats[i_mesh] = np.mean(ml_xhats, axis=0)
		data_set = 0
		for n_trial in range(N_data * N_trials):
			rmse = np.sqrt(np.mean((ref_xhats[data_set, :custm_length] - ml_xhats[n_trial, :custm_length]) ** 2))
			ml_rmse[i_mesh, n_alloc, n_trial] = rmse
			for n in range(length):
				rmse = np.sqrt(np.mean((ref_xhats[data_set, :n + 1] - ml_xhats[n_trial, :n + 1]) ** 2))
				var_ml_rmse[i_mesh, n_alloc, n_trial, n] = rmse
			if (n_trial + 1) % N_trials == 0:
				data_set += 1

# Compute the means over all data and all trials #
for i_mesh in range(N_MESHES):
	for n_alloc in range(alloc_counters[i_mesh]):
		mean_ml_rmse[i_mesh, n_alloc] = np.mean(np.log10(ml_rmse[i_mesh, n_alloc]))
		for n in range(length):
			var_mean_ml_rmse[i_mesh, n_alloc, n] = np.mean(np.log10(var_ml_rmse[i_mesh, n_alloc, :, n]))

# Find the stepwise minimums of the MLBPF RMSE #
temp_mins = np.empty(N_MESHES)
N1_mins = np.empty(length)
level0_mins = np.empty(length)
color_mins = np.empty(length, dtype=int)
for n in range(length):
	mins = []; maxs = []; N1_mins_ind = []; nx0_mins_t = [];
	for i_mesh in range(N_MESHES):
		mins.append(np.min(var_mean_ml_rmse[i_mesh, :alloc_counters[i_mesh], n]))
		maxs.append(np.max(var_mean_ml_rmse[i_mesh, :alloc_counters[i_mesh], n]))
		N1_mins_ind.append(np.argmin(var_mean_ml_rmse[i_mesh, :alloc_counters[i_mesh], n]))
	for i_mesh in range(N_MESHES):
		temp_mins[i_mesh] = var_mean_ml_rmse[i_mesh, N1_mins_ind[i_mesh], n]
	N1_mins[n] = N1s[N1_mins_ind[np.argmin(temp_mins)]]
	level0_mins[n] = level0s[np.argmin(temp_mins)]
	color_mins[n] = np.argmin(temp_mins)
	min_mean_ml_rmse[n] = np.min(mins)
	max_mean_ml_rmse[n] = np.max(maxs)

glob_min = np.min((np.min(min_mean_ml_rmse[:custm_length]), np.min(mean_bpf_rmse[:custm_length])))
glob_max = np.max((np.max(max_mean_ml_rmse[:custm_length]), np.max(mean_bpf_rmse[:custm_length])))
glob_min = glob_min - 0.15 * np.abs(glob_min); glob_max = glob_max + 0.15 * np.abs(glob_max)
# print(mean_bpf_rmse)
# print(min_mean_ml_rmse)
if pd.isna(glob_min):
	glob_min = -4.5
if pd.isna(glob_max):
	glob_max = -0.5



# ---------------------------------------------------- Plotting ----------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
colors = ["orchid", "mediumpurple", "royalblue", "powderblue", "mediumseagreen", "greenyellow", "orange", "tomato", "firebrick"]

# These are the sequential RMSE plots #
# ----------------------------------- #
for n in range(custm_length):
	fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
	ax1.plot(N1s[:max_allocs], mean_bpf_rmse[n] * np.ones(max_allocs), color="black", label="BPF")
	mins = []; N1_min = []; nx0_min = []
	temp_min = glob_max
	for i_mesh in range(N_MESHES):
		ax1.plot(N1s[:alloc_counters[i_mesh]], var_mean_ml_rmse[i_mesh, :alloc_counters[i_mesh], n], label=level0s[i_mesh], marker="o", markersize=3)
	ax1.set(ylim=(glob_min, glob_max))
	ax1.set_title("N_BPF = {}, n = {}".format(N_bpf, n))
	ax1.set_xlabel("N1")
	ax1.legend()
	plt.savefig("seq_RMSE_n={}_len={}.png".format(n, length))


# This is the sequence of ML RMSE mins wrt allocation and level 0 mesh #
# -------------------------------------------------------------------- #
# fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
# for n in range(custm_length):
# 	ax2.plot(N1_mins[n], n, marker="o", color=colors[color_mins[n]], label=level0_mins[n])
# ax2.set_xlabel(r"$N_1$")
# ax2.set_ylabel("n")
# ax2.legend()
# plt.savefig("sequential_mins_len=50.png")


# These are the min RMSE plots #
# ---------------------------- #
fig3, ax3 = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))
ax3[0].plot(range(custm_length), mean_bpf_rmse[:custm_length], color="black", label="BPF log10(RMSE)")
ax3[0].plot(range(custm_length), min_mean_ml_rmse[:custm_length], color="red", label="Optimal MLBPF log10(RMSE)")
ax3[0].set_xticks([])
ax3[0].legend()

ratio_min_rmses = mean_bpf_rmse / min_mean_ml_rmse
ax3[1].plot(range(custm_length), ratio_min_rmses[:custm_length], label="ML / BPF")
ax3[1].plot(range(custm_length), np.ones(custm_length), color="black")
ax3[1].set_xticks([])
ax3[1].legend()

sq_diffs = (min_mean_ml_rmse - mean_bpf_rmse) ** 2 / ref_stds
ax3[2].plot(range(custm_length), sq_diffs[:custm_length] * 100, marker="o", markersize=3)
ax3[2].set_xlabel("n")
ax3[2].set_ylabel("%")
# ax3[2].set_title(r"$(rmse_n(ML) - rmse_n(BPF))^2 / \sigma(ref_n)$") //// Do we sqrt?

fig3.suptitle("N(BPF) = {}".format(N_bpf))
# plt.savefig("min_rmse_len={}.png".format(length))


# These are the mean drift plots #
# ------------------------------ #
# fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
# ax4.plot(range(length), mean_ref_xhats, label="ref xhats", color="black")
# for i_mesh in range(N_MESHES):
# 	ax4.plot(range(length), N1_0_mean_ml_xhats[i_mesh], label=level0s[i_mesh], marker="o", markersize=3)
# ax4.set_title(r"Drift of ML xhats for $N_1=0$")
# ax4.set_xlabel("n")
# ax4.legend()
# plt.savefig("drift_plots.png")


# This is the final iterate RMSE plot #
# ----------------------------------- #
# fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
# ax1.plot(N1s[:max_allocs], mean_bpf_rmse * np.ones(max_allocs), color="black", label="BPF")
# for i_mesh in range(N_MESHES):
# 	ax1.plot(N1s[:alloc_counters[i_mesh]], mean_ml_rmse[i_mesh, :alloc_counters[i_mesh]], label=level0s[i_mesh], marker="o", markersize=3)
# 	print(mean_ml_rmse[i_mesh, :alloc_counters[i_mesh]])
# ax1.set_title("Mean log10(RMSE) at n = {}".format(custm_length - 1))
# ax1.legend()

plt.show()



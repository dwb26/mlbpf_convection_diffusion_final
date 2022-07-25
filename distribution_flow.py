import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

bpf_distr_f = open("bpf_distr.txt", "r")
ref_particles_f = open("ref_particles.txt", "r")
res_bpf_distr_f = open("res_bpf_distr.txt", "r")
res_ref_distr_f = open("res_ref_distr.txt", "r")

length, N_bpf = list(map(int, bpf_distr_f.readline().split()))
length, N_ref = list(map(int, ref_particles_f.readline().split()))

bpf_x = np.empty(length * N_bpf)
bpf_w = np.empty(length * N_bpf)
ref_x = np.empty(length * N_ref)
ref_w = np.empty(length * N_ref)

for n in range(length * N_bpf):
	bpf_x[n], bpf_w[n] = list(map(float, bpf_distr_f.readline().split()))
for n in range(length * N_ref):
	ref_x[n], ref_w[n] = list(map(float, ref_particles_f.readline().split()))
res_bpf_distr = np.array(list(map(float, res_bpf_distr_f.readline().split()))).reshape((length, N_bpf))
res_ref_distr = np.array(list(map(float, res_ref_distr_f.readline().split()))).reshape((length, N_ref))

bpf_x = np.reshape(bpf_x, (length, N_bpf))
bpf_w = np.reshape(bpf_w, (length, N_bpf))
ref_x = np.reshape(ref_x, (length, N_ref))
ref_w = np.reshape(ref_w, (length, N_ref))

# bpf_xmin = np.min(bpf_x); bpf_xmax = np.max(bpf_x)
bpf_xmin = np.min(res_bpf_distr); bpf_xmax = np.max(res_bpf_distr)
bpf_ymin = np.min(bpf_w); bpf_ymax = np.max(bpf_w)
ref_xmin = np.min(ref_x); ref_xmax = np.max(ref_x)
xmin = np.min((bpf_xmin, ref_xmin))
xmax = np.min((bpf_xmax, ref_xmax))

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
# axs.hist(bpf_x[0], 50, density=True, facecolor='g', alpha=0.75, weights=bpf_w[0])
# axs.hist(res_bpf_distr[0], 50, density=True, facecolor='g', alpha=0.75)

# for n in range(length - 1, -1, -1):
for n in range(length):
	fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
	# axs.hist(x=bpf_x[n], bins=50, density=True, weights=bpf_w[n])
	axs.hist(res_bpf_distr[n], 50, density=True, facecolor='g', alpha=0.75)
	axs.hist(res_ref_distr[n], 50, density=True, facecolor='r', alpha=0.75)
	axs.set(xlim=(xmin, xmax))
plt.show()
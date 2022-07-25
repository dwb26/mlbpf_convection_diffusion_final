import numpy as np
import matplotlib.pyplot as plt

bpf_xhats_f = open("bpf_xhats.txt", "r")
# ml_xhats_f = open("ml_xhats.txt", "r")
full_hmm_data_f = open("full_hmm_data.txt", "r")
ref_xhats_f = open("ref_xhats.txt", "r")
nx0 = 5; N1 = 250; n_data = 0
ml_xhats_f = open("raw_ml_xhats_nx0={}_N1={}_n_data={}.txt".format(nx0, N1, n_data), "r")

ref_xhats = list(map(float, ref_xhats_f.readline().split()))
length = len(ref_xhats)
signal = np.empty(length); observations = np.empty(length)
for n in range(length):
	signal[n], observations[n] = list(map(float, full_hmm_data_f.readline().split()))	
bpf_xhats = np.array(list(map(float, bpf_xhats_f.readline().split())))
# ml_xhats = np.array(list(map(float, ml_xhats_f.readline().split())))

fig = plt.figure(figsize=(16, 8))
ax = plt.subplot(111)
ax.scatter(range(length), signal, s=3, label="signal")
ax.scatter(range(length), ref_xhats, s=3, label="ref bpf")
ax.scatter(range(length), bpf_xhats, s=3, label="bpf")
# ax.scatter(range(length), ml_xhats, s=3, label="mlbpf")
ax.set_xlabel("$n$-th iterate", fontsize=16)
ax.set_ylabel(r"$\widehat{\theta}_n$", fontsize=16)
ax.legend()
plt.suptitle("Mean estimates for $N = {}$ BPF particles, nx0 = {}".format(2500, 250), fontsize=16)
plt.tight_layout()
plt.show()
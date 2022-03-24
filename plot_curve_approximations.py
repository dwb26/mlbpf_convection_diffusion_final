import numpy as np
import matplotlib.pyplot as plt

nx1 = 50; length = 5; space_left = 0.0; space_right = 1.0; Ns_fine = 2000; N1 = 3
nx0s = [20, 10, 5]
corrections = np.empty((length, N1))
corrections_x = np.empty((length, N1))
regression_data = np.empty((length, Ns_fine))
regression_data_x = np.empty((length, Ns_fine))
exact_corrections = np.empty((length, Ns_fine))
exact_corrections_x = np.empty((length, Ns_fine))
corrections_temp = np.empty((N1 * length, 2))
exact_corrections_temp = np.empty((Ns_fine * length, 2))

fig, axs = plt.subplots(nrows=len(nx0s), ncols=length, figsize=(15, 8))

z = 0
for nx0 in nx0s:
	regression_data_f = open("regression_data_nx0={}.txt".format(nx0), "r")
	corrections_f = open("corrections_nx0={}.txt".format(nx0), "r")
	exact_corrections_f = open("exact_corrections_nx0={}.txt".format(nx0), "r")	
	
	for k in range(Ns_fine * length):
		exact_corrections_temp[k] = list(map(float, exact_corrections_f.readline().split()))
	for k in range(N1 * length):
		corrections_temp[k] = list(map(float, corrections_f.readline().split()))
	exact_corrections_x = exact_corrections_temp[:, 0].reshape((length, Ns_fine))
	exact_corrections = exact_corrections_temp[:, 1].reshape((length, Ns_fine))
	corrections_x = corrections_temp[:, 0].reshape((length, N1))
	corrections = corrections_temp[:, 1].reshape((length, N1))

	m = 0; n = 0
	for k in range(2 * length):
		if k % 2 == 0:
			regression_data_x[m] = list(map(float, regression_data_f.readline().split()))
			m += 1
		else:
			regression_data[n] = list(map(float, regression_data_f.readline().split()))
			n += 1


	l1_err = 0
	for n in range(length):
		l1_err += np.sum(np.abs(regression_data[n] - exact_corrections[n]))

	# fig, axs = plt.subplots(nrows=1, ncols=length, figsize=(15, 8))
	for k in range(length):
		axs[z, k].plot(exact_corrections_x[k], exact_corrections[k], label="true correction curve", color="black")
		axs[z, k].plot(regression_data_x[k], regression_data[k], label="regressed correction curve")
		axs[z, k].scatter(corrections_x[k], corrections[k], label="correction data", s=2, color="red")
		if k == length - 1 and z == 0:
			axs[z, k].legend()
		axs[z, 0].set_ylabel("nx0 = {}".format(nx0))
		if z == 0:
			axs[z, k].set_title("n = {}".format(k))
		if k == length - 1:
			ax = axs[z, k].twinx()
			ax.set_ylabel("L1 err: {:.1e}".format(l1_err))
		if k == 0 and z == 2:
			print(regression_data[k])

	z += 1

	# plt.suptitle("Correction data for nx0 = {}, total l1 error = {:e}".format(nx0, l1_err))
	# plt.savefig("nx0={}_corrections.png".format(nx0))
	# plt.show()

	regression_data_f.close()
	corrections_f.close()
	exact_corrections_f.close()

plt.suptitle("Correction data and aggregate l1 error for initial signal value = -3 ")
plt.savefig("s0=-3_correction_plots.png".format(nx0))
fig.subplots_adjust(hspace=0.6)
plt.show()
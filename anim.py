import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

hmm_f = open("hmm_data_n_data=0.txt", "r")
curve_data_f = open("curve_data.txt", "r")
curve_data0_f = open("curve_data0.txt", "r")

length = int(hmm_f.readline())
signal = np.empty(length)
sig_sd, obs_sd = list(map(float, hmm_f.readline().split()))
space_left, space_right = list(map(float, hmm_f.readline().split()))
nx = 50; nt = 25
nx0 = 10; nt0 = 25
curves = np.empty((length * nt, nx))
curves0 = np.empty((length * nt0, nx0))

m = 0; n = 0
for line in curve_data0_f:
	curves0[m] = list(map(float, line.split()))
	m += 1
for line in curve_data_f:
	curves[n] = list(map(float, line.split()))
	n += 1

fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
xs = np.linspace(space_left, space_right, nx)
xs0 = np.linspace(space_left, space_right, nx0)
line, = ax.plot(xs, curves[0])
line0, = ax.plot(xs0, curves0[0])

def update(n):
	line.set_data(xs, curves[n])
	line0.set_data(xs0, curves0[n])
	# ax.set_title("iterate = {} / {}".format(n, total_length))
	ax.set(ylim=(np.min(curves), np.max(curves)))
	return line, line0,
	# return line,

ani = animation.FuncAnimation(fig, func=update, frames=range(1, length * nt, 1))
plt.show()
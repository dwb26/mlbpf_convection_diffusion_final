import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

true_curves_f = open("true_curves.txt", "r")
ml_curves_f = open("ml_curves.txt", "r")
ml_curves0_f = open("ml_curves0.txt", "r")
ml_mesh_f = open("ml_mesh.txt", "r")
ml_mesh0_f = open("ml_mesh0.txt", "r")

nx = 50
nx0 = 10
length = 8

true_curves = np.empty((length, nx))
ml_curves = np.empty((length, nx))
ml_curves0 = np.empty((length, nx0))
ml_mesh = np.empty((length, nx))
ml_mesh0 = np.empty((length, nx0))
for n in range(length):
	true_curves[n] = list(map(float, true_curves_f.readline().split()))
	ml_curves[n] = list(map(float, ml_curves_f.readline().split()))
	ml_curves0[n] = list(map(float, ml_curves0_f.readline().split()))
	ml_mesh[n] = list(map(float, ml_mesh_f.readline().split()))
	ml_mesh0[n] = list(map(float, ml_mesh0_f.readline().split()))

fig, ax = plt.subplots(2, 4, figsize=(10, 7))
k = 0
for m in range(length):
	ax[k, m % 4].plot(ml_mesh[m], true_curves[m], label="true")
	ax[k, m % 4].plot(ml_mesh[m], ml_curves[m], label="ml1")
	ax[k, m % 4].plot(ml_mesh0[m], ml_curves0[m], label="ml0")
	if (m + 1) % 4 == 0:
		k += 1
ax[0, 0].legend()
plt.show()
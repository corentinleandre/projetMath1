# Python code to generate random numbers from multivariate Gaussia
import scipy.stats as stats
from matplotlib import pyplot as plt
import numpy as np

mu = [0, 0]
Sigma = [[0.25, 0.3], [0.3, 1.0]]
x = stats.multivariate_normal.rvs(mu, Sigma, 1000)
plt.scatter(x[:, 0], x[:, 1], edgecolors="black")

levels = [0.01/(2 * np.pi), 0.05/(2 * np.pi), 0.95/(2 * np.pi)]

# creation de Xpos , matrice de coordonnees des points du plan
# au centieme
x1 = np.arange(-3.5, 3.5, 0.01) # valeurs en abscisses
x2 = np.arange(-3.5, 3.5, 0.01) # valeurs en ordonnees
X1, X2 = np.meshgrid(x1, x2)
Xpos = np.empty(X1.shape + (2,))
Xpos[:, :, 0] = X1
Xpos[:, :, 1] = X2
# fin Xpos

f = stats.multivariate_normal.pdf(Xpos, mu, Sigma)
plt.gca().set_aspect('equal', adjustable='box')
#plt.scatter(z[:, 0], z[:, 1], edgecolors="black")

plt.contour(x1, x2, f, levels, colors="red")
plt.show()
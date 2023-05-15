# Python code to generate random numbers from multivariate Gaussia
import scipy.stats as stats
from matplotlib import pyplot as plt

mu = [0, 0]
Sigma = [[0.25, 0.3], [0.3, 1.0]]
#x = stats.multivariate_normal.rvs(mu, Sigma, 10000)
#plt.scatter(x[:, 0], x[:, 1], edgecolors="black")
#plt.show()

# Python code to generate random numbers from MV Gaussian
# with the Gaussian contour .
import numpy as np
import scipy . stats as stats
import matplotlib . pyplot as plt
z = stats.multivariate_normal.rvs(mu, Sigma, 1000)
x1 = np.arange(-3.5, 3.5, 0.01) # valeurs en abscisses
x2 = np.arange(-3.5, 3.5, 0.01) # valeurs en ordonnees
# creation de Xpos , matrice de coordonnees des points du plan
# au centieme
X1, X2 = np.meshgrid(x1, x2)
Xpos = np.empty(X1.shape + (2,))
Xpos[:, :, 0] = X1
Xpos[:, :, 1] = X2
# fin Xpos
f = stats.multivariate_normal.pdf(Xpos, mu, Sigma)
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(z[:, 0], z[:, 1], edgecolors="black")
levels = [0.01/(2 * np.pi), 0.05/(2 * np.pi), 0.95/(2 * np.pi)]
plt.contour(x1, x2, f, levels, colors="red")

# Estimateurs empiriques
muEmp = [np.mean(z[:, 0]), np.mean(z[:, 1])]
#SigmaEmp = [[np.var(z[:, 0]), np.cov(z)], [np.cov(z), np.var(z[:, 1])]]
SigmaEmp = np.cov(z, rowvar=False)
#print(muEmp)
#print(SigmaEmp)

g = stats.multivariate_normal.pdf(Xpos, muEmp, SigmaEmp)
plt.contour(x1, x2, g, levels, colors="green")
plt.show()



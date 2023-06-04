# Python code to generate random numbers from multivariate Gaussia
import scipy.stats as stats
from matplotlib import pyplot as plt
import numpy as np

# - constants
mu = [0, 0]
Sigma = [[1, 0], [0, 1.0]]

# - generating points
x = stats.multivariate_normal.rvs(mu, Sigma, 1000)
plt.scatter(x[:, 0], x[:, 1], edgecolors="black")

# - making Xpos , matrix of plane points
# within a hundredth
x1 = np.arange(mu[0]-3.5, mu[0]+3.5, 0.01) # valeurs en abscisses
x2 = np.arange(mu[1]-3.5, mu[1]+3.5, 0.01) # valeurs en ordonnees
X1, X2 = np.meshgrid(x1, x2)
Xpos = np.empty(X1.shape + (2,))
Xpos[:, :, 0] = X1
Xpos[:, :, 1] = X2
# end Xpos

# - probability wanted
p = [0.01, 0.5, 0.95]

# - treating for function
p.sort(reverse=True)
levels = [(1-x)/(2*np.pi) for x in p]

# - density function and it's ellipses from given constants
f = stats.multivariate_normal.pdf(Xpos, mu, Sigma)
plt.gca().set_aspect('equal', adjustable='box')
#plt.scatter(z[:, 0], z[:, 1], edgecolors="black")

plt.contour(x1, x2, f, levels, colors="red")

plt.show()

# Python code to generate random numbers from MV Gaussian
# with the Gaussian contour .
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# - constants
mu = [0, 0]
Sigma = [[1, 0], [0, 1.0]]
levels = [0.01/(2 * np.pi), 0.05/(2 * np.pi), 0.95/(2 * np.pi)]

# - making Xpos , matrix of plane points
# within a hundredth
x1 = np.arange(mu[0]-3.5, mu[0]+3.5, 0.01)  # abscissa values
x2 = np.arange(mu[1]-3.5, mu[1]+3.5, 0.01)  # ordinate values
X1, X2 = np.meshgrid(x1, x2)
Xpos = np.empty(X1.shape + (2,))
Xpos[:, :, 0] = X1
Xpos[:, :, 1] = X2
# end Xpos

plt.subplots(3, 2)
plt.title("diagrammes représentant les isodensités pour certains n")
fig = plt.gcf()
fig.set_size_inches(18, 12)
for i in range(1, 7):
    # - plt setup
    print(i)
    ax = plt.subplot(3, 2, i)
    plt.gca().set_aspect('equal', adjustable='box')

    # - generating points
    z = stats.multivariate_normal.rvs(mu, Sigma, np.power(10,i))
    # plt.scatter(z[:, 0], z[:, 1], edgecolors="black")

    # - density function and it's ellipses from given constants
    f = stats.multivariate_normal.pdf(Xpos, mu, Sigma)
    plt.contour(x1, x2, f, levels, colors="red")

    # - empirical estimates
    muEmp = [np.mean(z[:, 0]), np.mean(z[:, 1])]
    SigmaEmp = np.cov(z, rowvar=False)
    # print(muEmp)
    # print(SigmaEmp)

    # - density function and it's ellipses from empirical estimators
    g = stats.multivariate_normal.pdf(Xpos, muEmp, SigmaEmp)
    plt.contour(x1, x2, g, levels, colors="green")

    # - labeling plots
    plt.title('n = ' + str(np.power(10, i)))
    red_line = mlines.Line2D([], [], color='red', label='constants')
    green_line = mlines.Line2D([], [], color='green', label='estimators')
    ax.legend(handles=[red_line, green_line ])
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fourier import FourierBasis

#creating training data
f =  lambda x,y : (3.*(1.-x)**2.*np.exp(-(x**2) - (y+1.)**2.)
    - 10.*(x/5. - x**3 - y**5)*np.exp(-x**2-y**2)
    - 1./3.*np.exp(-(x+1.)**2 - y**2))

#2d input grid
Xs,Ys = np.meshgrid(np.linspace(-3,3),np.linspace(-3,3))
#target outputs
Zs = f(Xs,Ys)

#fourier basis
rep = FourierBasis(low=np.array([-3,-3.]),high=np.array([3.,3.]),
                    order = 6)
print rep.coeffs

#stochastic gradient descent
theta = np.zeros(rep.n_features)
alpha = 0.0001
for i in range(5000000):
    #pick random sample
    idx = np.random.randint(Zs.size)
    x,y,z = Xs.flat[idx],Ys.flat[idx],Zs.flat[idx]
    #get representation
    phi = rep.phi(np.array([x,y]))
    #get prediction
    pred = np.dot(phi,theta)
    #gradient update
    theta -=  alpha*0.5*(pred-z)*phi

#plot result
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(Xs, Ys, Zs)
ax.set_title('ground truth')
ax.set_zlim(-5, 5)

Ps = np.zeros_like(Zs)
for idx in range(Zs.size):
    Ps.flat[idx] = np.dot(rep.phi(np.array([Xs.flat[idx],Ys.flat[idx]])),theta)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(Xs, Ys,Ps)
ax.set_title('predictions')
ax.set_zlim(-5, 5)

plt.show()

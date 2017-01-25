import numpy as np
import matplotlib.pyplot as plt
from fourier import FourierBasis

'''
This file plots example basis functions from 6th order fourier basis over
2 state variables. See http://lis.csail.mit.edu/pubs/konidaris-aaai11a.pdf
'''

# generate 2D grid over [-1,1] x [-1,1]
Xs,Ys = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))


# create fourier basis
rep = FourierBasis(low=np.array([-1,-1.]),high=np.array([1.,1.]),
                    order = 6)
print rep.coeffs

#get representation of input data
bf = np.zeros((Xs.size,rep.n_features))
for idx in range(Xs.size):
    x,y = Xs.flat[idx],Ys.flat[idx]
    bf[idx,] = rep.phi(np.array([x,y]))

# plot subset of basis functions, title is coeffs
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
axes[0,0].matshow(bf[:,0].reshape(Xs.shape).T,cmap='gray')
axes[0,0].set_title(str(rep.coeffs[0,:]))
axes[0,1].matshow(bf[:,1].reshape(Xs.shape).T,cmap='gray')
axes[0,1].set_title(str(rep.coeffs[1,:]))
axes[0,2].matshow(bf[:,6].reshape(Xs.shape).T,cmap='gray')
axes[0,2].set_title(str(rep.coeffs[6,:]))
axes[1,0].matshow(bf[:,7].reshape(Xs.shape).T,cmap='gray')
axes[1,0].set_title(str(rep.coeffs[7,:]))
axes[1,1].matshow(bf[:,5].reshape(Xs.shape).T,cmap='gray')
axes[1,1].set_title(str(rep.coeffs[5,:]))
axes[1,2].matshow(bf[:,11].reshape(Xs.shape).T,cmap='gray')
axes[1,2].set_title(str(rep.coeffs[11,:]))

for ax in axes.flatten():
    ax.set_xticklabels([])
    ax.set_yticklabels([])
plt.show()

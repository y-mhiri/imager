#%%

from copy import deepcopy

import click
import h5py
import os 

import numpy as np
import matplotlib.pyplot as plt
from cmastro import *
from utils import *
from PIL import Image
from copy import deepcopy
from imager import *

rng = np.random.default_rng(0)

def generate_data():

    nbaselines = 128
    nsamples = 100

    ## Load Image

    npix = 64
    I = np.array(Image.open('../data/IMG/M31.jpeg').convert("L").resize((npix, npix)))
    x = vec(I)


    ## Generate visibility mask and Fourier Matrix

    dx, dy = 0.0125, 0.0125 #direction cosine
    srx, sry = npix/(2*dx), npix/(2*dy) #sample rate
    wl = 0.1 #wavelength
    uvpixsize = srx/(npix*wl) # en longueur d'ondes

    uvplane, vis = get_uvw("../data/MS/simulatedSky.ms")
    umax, vmax = uvpixsize * npix/2, uvpixsize * npix/2

    S = np.zeros((nbaselines, npix**2))
    mask = np.zeros((npix, npix)).astype(complex)
    sigma = 10
    bidx = 0
    temp = []
    uv_idx = np.linspace(-umax, umax, npix)
    for uvw, vi in zip(uvplane, vis):
        uidx = np.argmin(np.abs(uvw[0]/wl - uv_idx))
        vidx = np.argmin(np.abs(uvw[1]/wl - uv_idx))
        if not (uidx, vidx) in temp:
            mask[uidx, vidx] = 1
            S[bidx, vidx*npix + uidx] = 1
            bidx += 1
        if bidx >= nbaselines:
            break

        temp.append((uidx, vidx))

    Fmat = fourier_matrix(npix, npix)
    F = np.kron(Fmat, Fmat)
    A = S @ F
    Ainv = np.linalg.pinv(A)


    #%%

    #%%
    ## Generate data 


    I = A@x 
    P0 = np.linalg.norm(I)**2


    rho = 4
    ratio = 0.5
    ncontaminated = int(ratio*nbaselines)
    idxs = rng.permutation(np.arange(nbaselines))
    Pi = np.linalg.norm(I[idxs[0:ncontaminated]])**2

    return Ainv, S, F, Pi, P0, idxs, rho, x, I, npix, nbaselines, ncontaminated


# %%




def monte_carlo_step(idxs, nbaselines, ncontaminated, rho, T, I, x, npix, Ainv, S, F, SNR_RFI, Pi, SNR, P0, mc=0, path=".", rng=np.random.RandomState(0)):

    y = np.zeros((T,nbaselines,1)).astype(complex)

    sigma2 = 10**(-SNR/10)*P0/nbaselines

    sigmarfi = 10**(-SNR_RFI/10)*Pi
    W0[idxs[0:ncontaminated]] = np.ones(rho)
    W0 = W0/np.linalg.norm(W0)
    W = np.sqrt(sigmarfi) * W0

    
    
        
    print(f"Monte carlo step : {mc+1}")
    print(f"SNR RFI : {SNR_RFI}")

    for t in range(T):
        n = fast_complex_normal(np.zeros_like(I), sigma2*np.eye(nbaselines))
    # c = fast_complex_normal(vec(np.eye(int(np.sqrt(rho)))), np.eye(rho))
        c = fast_complex_normal(np.zeros(rho), np.eye(rho))
        y[t] = I + n + W@c


    xdirty = Ainv@np.sum(y, axis=0)/T

    print(f"Imaging... Gaussian model")
    xgauss, sigmagauss = emimager(y, xdirty, sigma2, 100, F, S, npix, nu=10, alpha=0.005, model='gaussian' )
    print(f"Imaging... Student model")
    xst, sigmast = emimager(y, xdirty, sigma2, 100, F, S, npix, nu=10, alpha=0.005, model='student' )
    # xst, sigmast = emimager(y, xdirty, sigma2, 10, F, S, npix, nu=5, alpha=0.01, model='student' )

    err = np.zeros(2)
    err[0] = np.linalg.norm(xgauss - x)**2/np.linalg.norm(x)**2
    err[1] = np.linalg.norm(xst - x)**2/np.linalg.norm(x)**2

    
    ## Save results
    with h5py.File(path,'a') as file:

        
        file.create_group(f"MC_{mc}")
        dset = file[f"MC_{mc}"]

        dset.create_dataset(f"Error", data=err)
        dset.create_dataset(f"Images/student", data=np.abs(xst).reshape(npix,npix, order='F'))

        dset.create_dataset(f"Images/gaussian", data=np.abs(xgauss).reshape(npix,npix, order='F'))
        dset.create_dataset(f"Images/dirtyimage", data=np.abs(xdirty).reshape(npix,npix, order='F'))
        dset.create_dataset(f"vis", data=y)



# %%
@click.command()
@click.option('--SNR_RFI', default=0)
@click.option('--SNR', default=10)
def main(snr, snr_rfi):
    Ainv, S, F, Pi, P0, idxs, rho, x, I, npix, nbaselines, ncontaminated = generate_data()

    T = 1
    mc = os.getenv("SLURM_PROCID")
    rng = np.random.default_rng()

    monte_carlo_step(idxs, nbaselines, ncontaminated, rho, T, I, x, npix, Ainv, S, F, snr_rfi, Pi, snr, P0, mc=mc, rng=rng)

    return True


if __name__ == '__main__':

    main()
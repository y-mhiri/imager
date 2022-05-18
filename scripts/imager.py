import numpy as np
import scipy as sc
from copy import deepcopy
from casacore.tables import table
from utils import *
from scipy.sparse import csr_matrix

global lightspeed 
lightspeed = 3.0e8


class Imager():

    def __init__(self, msfile, cellsize, npix):

        """
        msfile : path of a valid measurement set file
        cellsize : size of a pixel in arcsecond
        width : dimension of the image in pixel
        
        """
        self.t = table(msfile)
        self.nvis = len(self.t)
        
        self.antenna_table = table("{}/ANTENNA".format(msfile))
        self.antenna_pos = np.array([antenna['POSITION'] for antenna in self.antenna_table])
        self.nantenna = len(self.antenna_table)
        self.nbaselines = len(self.t) - self.nantenna

        self.tf = table("{}/SPECTRAL_WINDOW".format(msfile))
        self.freqs = [band["REF_FREQUENCY"] for band in self.tf]

        self.uvplane = np.array([[telem['UVW']] for telem in self.t]).reshape((-1, 3))
        self.vis = np.array([[telem['DATA']] for telem in self.t])
        

        self.cellsize = cellsize
        self.npix = npix
        self.umax, self.vmax = (1/(2*cellsize)), (1/(2*cellsize))

        Fmat = fourier_matrix(npix, npix)
        self.F = np.kron(Fmat, Fmat)

    def extract_data(self, frequency_band):

        wl = lightspeed/self.freqs[frequency_band]
        
        #S = np.zeros((nbaselines, self.npix**2))
        # mask = np.zeros((self.npix, self.npix)).astype(complex)

        bidx = 0
        uv_idx = np.linspace(-self.umax, self.umax, self.npix)

        row, col = np.zeros(self.nbaselines), np.zeros(self.nbaselines)
        data = np.zeros(self.nbaselines)
        
        y = np.zeros(self.nbaselines)
        for vi in self.t:

            uvw = vi["UVW"]
            p, q = vi["ANTENNA1"], vi["ANTENNA2"]
            if p<q:
            
                uidx = np.argmin(np.abs(uvw[0]/wl - uv_idx))
                vidx = np.argmin(np.abs(uvw[1]/wl - uv_idx))
                # mask[uidx, vidx] = 1
                # S[bidx, vidx*self.npix + uidx] = 1
                
                row[bidx] = bidx
                col[bidx] = int(vidx*self.npix + uidx)
                data[bidx] = 1

                y[bidx] = vi["DATA"][frequency_band, 0]
                bidx += 1
                if bidx%100:
                    print(f"baseline processed {bidx}")

        S = csr_matrix((data,(row, col)), shape=(self.nbaselines, self.npix**2))
        return S, y


    def emimager(self, sigma0, frequency_band, alpha=1, niter=10):
        
        
        #y = self.vis[:,frequency_band,0,0].reshape(-1,1)
        print(f"number of baseline : {self.nbaselines}")

        print("Creating mask matrix...")
        S, y = self.extract_data(frequency_band)
        print("Mask created")
        print("Computing the forward operator...")
        H = S@ self.F

        print("Dirty image inversion...")
        x0 = self.F.T.conj()/self.npix**2 @ S.T @y
        dirtyimage = np.abs(x0).reshape(self.npix,self.npix, order='F')
       
        # print("EM algorithm started...")
        # xk = deepcopy(x0) 
        # sigmak = sigma0

        # T, N  = 1, y.shape[0] 

        # z = np.zeros((T, x0.shape[0], 1)).astype(complex)
        # L = np.zeros(niter).astype(complex)
        # for i in range(niter):
        #     print(f"iteration {i}")
        #     ## Compute likelihood
        #     # mean, Cov = H@xk, sigmak*np.eye(N)
        #     # sign, logdet = np.linalg.slogdet(Cov)
        #     # l =  np.sum([logdet + (yt - mean).T.conj() @ np.linalg.solve(Cov, yt - mean)
        #     #                                                     for yt in y], axis=0)
        #     # L[i] = l.reshape(-1)[0]

        #     ## EM procedure
        #     for t in range(T):
        #         z[t] = self.F@xk + S.T @ (y[t] - H@xk)

        #     xk = self.F.T.conj()/self.npix**2 @ np.sum([zt for zt in z], axis=0)/T
        #     xk = np.sign(xk) * np.max([np.abs(xk)- alpha, np.zeros(xk.shape)], axis=0)

        #     sigmak = (1/(T*N)) * np.sum([np.linalg.norm(y[t] - H@xk)**2 for t in range(T)], axis=0)
        
        # image = np.abs(xk).reshape(self.npix,self.npix, order='F')
        
        return dirtyimage, #image, sigmak,L


def emimager(y, x0, sigma0, niter, F, S, npix, nu=0, alpha=0.01, model='gaussian'):

    A = S@F

    if model=='gaussian':
        
        xk = deepcopy(x0) 
        sigmak = sigma0

        T, N  = y.shape[0], y.shape[1]

        z = np.zeros((T, x0.shape[0], 1)).astype(complex)
        L = np.zeros(niter).astype(complex)
        for i in range(niter):
            
            ## Compute likelihood
            # mean, Cov = A@xk, sigmak*np.eye(N)
            # sign, logdet = np.linalg.slogdet(Cov)
            # l =  np.sum([logdet + (yt - mean).T.conj() @ np.linalg.solve(Cov, yt - mean)
            #                                                     for yt in y], axis=0)
            # L[i] = l.reshape(-1)[0]

            ## EM procedure
            for t in range(T):
                z[t] = F@xk + S.T @ (y[t] - A@xk)

            xk = F.T.conj()/npix**2 @ np.sum([zt for zt in z], axis=0)/T
            xk = np.sign(xk) * np.max([np.abs(xk)- alpha, np.zeros(xk.shape)], axis=0)

            sigmak = (1/(T*N)) * np.sum([np.linalg.norm(y[t] - A@xk)**2 for t in range(T)], axis=0)
            
    elif model=='student':

        xk = deepcopy(x0) 
        sigmak = sigma0

        T, N  = y.shape[0], y.shape[1]

        z = np.zeros((T, x0.shape[0], 1)).astype(complex)
        L = np.zeros(niter).astype(complex)

        tau = np.zeros((T, N))
        for i in range(niter):
            
            ## Compute likelihood
            # mean, Cov = A@xk, sigmak*np.eye(N)
            # sign, logdet = np.linalg.slogdet(Cov)
            # l =  np.sum([logdet + (yt - mean).T.conj() @ np.linalg.solve(Cov, yt - mean)
            #                                                     for yt in y], axis=0)
            # L[i] = l.reshape(-1)[0]

            ## EM procedure
            for t in range(T):
                tau[t] = (nu + 1)/(nu + np.linalg.norm(y[t] - A@xk, axis=1)) 
                z[t] = F@xk + S.T @ np.diag(tau[t]) @ (y[t] - A@xk)

            xk = F.T.conj()/npix**2 @ np.sum([zt for zt in z], axis=0)/T
            xk = np.sign(xk) * np.max([np.abs(xk)- alpha, np.zeros(xk.shape)], axis=0)

            sigmak = (1/(T*N)) * np.sum([(y[t] - A@xk).T.conj() @np.diag(tau[t])@(y[t]-A@xk)
                                                            for t in range(T)], axis=0)

    return xk, sigmak
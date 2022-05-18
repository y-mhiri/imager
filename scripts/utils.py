import numpy as np
from scipy.linalg import khatri_rao
from casacore.tables import table
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

def vec(M):
    return M.reshape((-1,1), order='F')
def unvec(M, shape):
    return M.reshape(shape)


def fourier_matrix(N, F):
    n = np.linspace(0, N-1, N).reshape((1, -1))
    f = np.linspace(-F//2, F//2-1, F).reshape((-1, 1))
    
    return np.exp(-1j*2*np.pi*n*f/N)


def sinimage(npix, dx,dy,fx,fy,wl):
    x, y = np.linspace(-dx,dx, npix), np.linspace(-dy,dy, npix)
    X, Y = np.meshgrid(x,y)
    I = np.sin((2*np.pi) * (X*fx*wl + Y*fy*wl))
    return I
    
def generate_directions(npix, theta=90):

    
    """
    generate direction cosines vectors (l,m,n) 
    for an image of size npix and field caracterized by its declination theta

    Input : 
    - (int) npix : image pixel size
    - (float) theta : declination angle from the Zenith between 0 and 90 degree

    Returns : lmn
    """

    lin = np.linspace(-1, 1, npix)
    X, Y = np.meshgrid(lin, lin)

    alpha = 90 - theta
    DEC = 90*(1 - np.sqrt(X**2 + Y**2))
    DEC = (1 - alpha/90)*DEC + alpha
    DEC = DEC*np.pi/180

    RA = np.arccos(Y/np.sqrt(X**2+ Y**2))
    RA[:,0:npix//2] =  -RA[:,0:npix//2] 
    
    L = np.cos(DEC)*np.cos(RA)
    M = np.cos(DEC)*np.sin(RA)
    N = np.sin(DEC)

    l = vec(L)
    m = vec(M)
    n = vec(N)
 
    lmn = np.array([l,m,n]).T

    return lmn

def em_imager(msfile, npix, wl, fov=(85,85)):
    """
    EM Imager 

    Input : 
    - (int) npix : image pixel size
    - (float) wl : wavelength
    - (float) theta : declination angle from the Zenith between 0 and 90 degree
    
    Returns : (ndarray) Image
    
    """
    # lmn = generate_directions(npix, theta)
    # lmn = lmn[0]


    antenna_pos = extract_antenna_position(msfile)
    nantenna = len(antenna_pos)
    nbaselines = int(nantenna*(nantenna+1)/2)
    f = get_ref_frequency(msfile)

    dx, dy = np.cos(fov[1]*np.pi/180)*np.cos(fov[0]*np.pi/180), np.cos(fov[1]*np.pi/180)*np.sin(fov[0]*np.pi/180)
    srx,sry = npix/(2*dx), npix/(2*dy)

    sampledFFT = np.zeros((npix, npix)).astype(complex)
    uvpixsize = srx/(npix*wl) # en longueur d'ondes
    umax, vmax = uvpixsize * npix/2, uvpixsize * npix/2

    uvw, vis = get_uvw(msfile)
    uvplane = uvw[:,0:2]
    idx = 0

    y = np.zeros(nbaselines)
    for idx, uv in enumerate(uvplane):
        uv_idx = np.linspace(-umax, umax, npix)
        uidx = np.argmin(np.abs(uv[0]/wl - uv_idx))
        vidx = np.argmin(np.abs(uv[1]/wl - uv_idx))
        sampledFFT[uidx, vidx] = vis[idx,0]
        y[idx] = vis[idx,0]

    
    return I_EM

def mvdr(R, pos, npix, wl, theta=0):

    """
    MVDR Beamforming 

    Input : 
    - (ndarray) R : Correlation matrix
    - (ndarray) pos : Antenna positions
    - (int) npix : image pixel size
    - (float) wl : wavelength
    - (float) theta : declination angle from the Zenith between 0 and 90 degree
    
    Returns : (ndarray) MVDR Image
    
    """
    
    lmn = generate_directions(npix, theta)
    lmn = lmn[0]
    A = np.exp(-1j*(2*np.pi/wl)* (pos) @ lmn.T)
    i_mvdr = np.array([(1/len(pos))**2 * a.T.conj() @ R @ a for a in A.T])
    I_MVDR = i_mvdr.reshape((npix, npix))

    return I_MVDR

def WLS_imager(R,pos, npix, wl, theta, SIGn = None, gain=None,W=None):

    """
    Weighted least square solution to get an image from a correlation matrix and a model
    specified by the antenna position, a gain matrix and a weight matrix


    Input : 
    - (ndarray) R : Correlation matrix
    - (ndarray) pos : Antenna positions
    - (int) npix : image pixel size
    - (float) wl : wavelength
    - (float) theta : declination angle from the Zenith between 0 and 90 degree
    - (ndarray) gain : gain matrix that accounts for systematic errors
    - (ndarray) SIGn : noise covariance matrix
    - (ndarray) W : weight matrix
    
    Returns : WLS Image
    
    """
    assert R.shape[0] == R.shape[1], "R not a square matrix"
    
    lmn = generate_directions(npix, theta)
    A = np.exp(-1j*(2*np.pi/wl)* (pos) @ lmn.T)
    G = np.eye(len(pos))
    Cov = np.zeros_like(R)
    if W==None:
        W = np.eye(len(R))
    
    assert W.shape == R.shape, "wrong shape for weight matrix"

    if gain!=None:
        assert gain.shape[0] == gain.shape[1], "gain not a square matrix"
        G = gain

    if SIGn!=None:
        assert SIGn.shape[0] == SIGn.shape[1], "SIGn not a square matrix"
        Cov = SIGn
    

    M = np.linalg.pinv(khatri_rao((W@G@A).conj(), W@G@A)) @ np.kron(W.conj(), W) 
    Iwls = M @ vec(R - Cov)

    return Iwls.reshape((npix, npix))

def extract_antenna_position(msfile, ignore_z=True):

    antenna_table = table("{}/ANTENNA".format(msfile))
    antenna_pos = np.array([antenna['POSITION'] for antenna in antenna_table])
    
    if ignore_z:
        antenna_pos[:,2] = 0*antenna_pos[:,2] 

    return antenna_pos

def get_ref_frequency(msfile):

    spectralWindow = table("{}/SPECTRAL_WINDOW".format(msfile))
    return spectralWindow[0]["REF_FREQUENCY"]


def get_uvw(msfile, plot=False):
    
    t = table(msfile)

    UVW = np.array([[telem['UVW']] for telem in t]).reshape((-1, 3))
    vis = np.array([[telem['DATA']] for telem in t]).reshape((-1, 4))
    if plot:

        plt.figure()
        plt.title('UV Plane')
        plt.plot(UVW[:, 0], UVW[:,1], '*')
        plt.xlabel('U')
        plt.ylabel('V')

        plt.show()
    
    return UVW, vis



def H(_M):
    return _M.conjugate().transpose()


def invert_diagonal(_M, _bloc_size):
    _shape = _M.shape
    if len(_shape) != 2:
        raise ValueError("Not a square Matrix")
    if _shape[0] != _shape[1]:
        raise ValueError("Not a square Matrix")
    if _shape[0] % _bloc_size != 0:
        raise ValueError("Invalid Bloc Size")

    _n_bloc = _shape[0] // _bloc_size

    _M_inv = np.zeros(_M.shape).astype(np.complex)
    for k in range(_n_bloc):
        _M_inv[_bloc_size * k:_bloc_size * (k + 1), _bloc_size * k:_bloc_size * (k + 1)] = \
            np.linalg.pinv(_M[_bloc_size * k:_bloc_size * (k + 1), _bloc_size * k:_bloc_size * (k + 1)])
    return _M_inv


def compute_log_likelihood(_data, distribution = "gaussian", **kwargs):

    if distribution == "gaussian":
        _mean, _Cov = kwargs["mean"] , kwargs["Cov"]
        sign, logdet = np.linalg.slogdet(_Cov)
        L =  logdet + (_data - _mean).T.conj() @ np.linalg.solve(_Cov, _data - _mean)

    if distribution == "student-t":
        _mean, _Cov, _nu = kwargs["mean"] , kwargs["Cov"], kwargs["nu"]
        p = len(_data)
        _, logdet = np.linalg.slogdet(_Cov)
        cste = gamma((_nu+p)/2)/((_nu*np.pi)**(p/2) * gamma(_nu/2)) 
        
        L = 0.5*logdet + np.log(cste * (1 + (_data - _mean).T.conj() @ np.linalg.solve(_Cov, _data - _mean)/_nu)**(-(_nu + p)/2))

    
    return L.reshape((-1))


def compute_mse(*args):

    MSE = np.zeros(len(args)).astype(np.complex)
    for i in range(len(args)):
        _x = args[i][1].reshape((-1, 1))
        _x_est = args[i][0].reshape((-1, 1))

        MSE[i] = np.linalg.norm(_x - _x_est)**2 / (np.linalg.norm(_x))**2
        #MSE[i] = np.sqrt(np.linalg.norm(_x - _x_est)**2 / np.product([ k for k in _x.shape]))

    return MSE


def fast_complex_normal(mean, Cov, diag=True, rng=np.random.RandomState(0)): 

        n_samples = len(mean)
        # Check if covariance matrix is symetric
        if not np.allclose(Cov,Cov.T.conjugate(), atol=1e-08):
            raise ValueError("Covariance matrix must be symetric.")

        SIGMA = np.zeros((n_samples*2, n_samples*2))
        SIGMA[0: n_samples, 0:n_samples] = np.real(Cov)
        SIGMA[n_samples: 2*n_samples, n_samples:2*n_samples] = np.real(Cov)
        
        SIGMA[0: n_samples, n_samples:2*n_samples] = -np.imag(Cov)
        SIGMA[n_samples: 2*n_samples, 0: n_samples] = np.imag(Cov)

        SIGMA = (1/2)*SIGMA

        if not diag:
            S,D,V = np.linalg.svd(SIGMA)

            # if not np.allclose(S,V.T.conjugate(), rtol=1):
            #     print(S - V.T.conjugate())
            #     raise ValueError("SVD - Covariance matrix is not symetric")

            S = np.dot(S, np.diag(np.sqrt(D)))

        else:
            S = np.sqrt(SIGMA)

        MU = np.zeros(2*n_samples)
        MU[0:n_samples] = np.real(mean).reshape(-1)
        MU[n_samples:2*n_samples] = np.imag(mean).reshape(-1)
        
        _y = np.dot(S , rng.normal(0, 1, 2*n_samples)) + MU

        return (_y[0:n_samples] + 1j*_y[n_samples::]).reshape((-1,1))
    
def kron_fast(A,B):
    a = A[:,np.newaxis,:,np.newaxis]
    a = A[:,np.newaxis,:,np.newaxis]*B[np.newaxis,:,np.newaxis,:]
    a.shape = (A.shape[0]*B.shape[0],A.shape[1]*B.shape[1])
    return a
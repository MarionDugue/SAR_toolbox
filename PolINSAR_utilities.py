# utilities.py
import numpy as np
from scipy.ndimage import uniform_filter

def HSV_colormap_to_rgb(colormap, h, s, v):
    """
    Makes an HSV-like RGB representation based on the given colormap instead
    of 'hsv' colormap.

    See https://en.wikipedia.org/wiki/HSL_and_HSV

    Parameters
    ----------
    colormap : function
        Colormap function. Takes the values in 'h' array and returns an RGBA
        value for each point. The ones in matplotlib.cm should be compatible
    h : ndarray
        Hue values. Usually between 0 and 1.0.
    s : ndarray
        Saturation values. Between 0 and 1.0.
    v : ndarray
        Value values. Between 0 and 1.0.

    Returns
    -------
    rgb: ndarray
        An array with the same shape as input + (3,) representing the RGB.
    """
    # Generate color between given colormap (colormap(h)) and white (ones)
    # according to the given saturation
    tmp = (1 - s)[..., np.newaxis] * np.ones(3) + s[..., np.newaxis] * colormap(h)[..., :3]
    # Scale it by value
    return v[..., np.newaxis] * tmp


def calculate_covariance(im1, im2, looksa, looksr):

     # ... apply definition
    corr = uniform_filter( np.real(im1*np.conj(im2)), [looksa, looksr] ) + \
        1j*uniform_filter( np.imag(im1*np.conj(im2)), [looksa, looksr] )

    # ... and back to main
    return corr

def calculate_eigenvalues_3(T11, T12, T13, T22, T23, T33):

    # Calculate and order (from max to min) the eigenvalues of a 3x3 hermitian matrix in closed-form.
    # Inputs can be 2D az - rg (rows - columns).

    # get dimensions
    dims = T11.shape

    # calculate auxiliary quantities
    A = T11*T22 + T11*T33 + T22*T33 - T12*np.conj(T12) - T13*np.conj(T13) - T23*np.conj(T23)
    B = T11**2 - T11*T22 + T22**2 -T11*T33 -T22*T33 + T33**2 + 3*T12*np.conj(T12) + 3*T13*np.conj(T13) + 3*T23*np.conj(T23)

    DET = T11*T22*T33 - T33*T12*np.conj(T12) - T22*T13*np.conj(T13) - T11*T23*np.conj(T23) + T12*np.conj(T13)*T23 + np.conj(T12)*T13*np.conj(T23)
    TR = T11 + T22 + T33
    Z = 27*DET-9*A*TR + 2*TR**3 + np.sqrt((27*DET-9*A*TR + 2*TR**3)**2-4*B**3)

    del DET

    # ... and here they are:
    LA = ( 1/3.*TR + 2**(1/3.)*B/(3*Z**(1/3.)) + Z**(1/3.)/(3*2**(1/3.)) )
    LB = ( 1/3.*TR - (1+1j*np.sqrt(3))*B/(3*2**(2/3.)*Z**(1/3.)) - (1-1j*np.sqrt(3))*Z**(1/3.)/(6*2**(1/3.)) )
    LC = ( 1/3.*TR - (1-1j*np.sqrt(3))*B/(3*2**(2/3.)*Z**(1/3.)) - (1+1j*np.sqrt(3))*Z**(1/3.)/(6*2**(1/3.)) )

    # now order them:
    dumm = np.zeros((dims[0], dims[1], 3), 'float32')
    dumm [:, :, 0] = np.real(LA)
    dumm [:, :, 1] = np.real(LB)
    dumm [:, :, 2] = np.real(LC)

    del LA, LB, LC

    L1 = np.max(dumm, axis = 2)
    L3 = np.min(dumm, axis = 2)
    L2 = np.sum(dumm, axis = 2) - L1 - L3

    del dumm

    return L1, L2, L3



def calculate_eigenvectors_3(T11, T12, T13, T22, T23, T33, L1, L2, L3) :

    # Calculate the eigenvectors corresponding to the eigenvalues (L1, L2, L3)
    # of a 3x3 matrix
    # Inputs can be 2D az - rg (rows - columns).

    # get dimensions
    dims = T11.shape

    # first eigenvector - corresponds to the maximum eigenvalue L1
    U1 = np.ones((dims[0], dims[1], 3), 'complex64')
    U1[:, :, 0] = (L1 -T33)/np.conj(T13) + (((L1-T33)*np.conj(T12) + np.conj(T13)*T23)*np.conj(T23))/ \
                    (((T22-L1)*np.conj(T13) - np.conj(T12)*np.conj(T23))*np.conj(T13))
    U1[:, :, 1] = -((L1-T33)*np.conj(T12)+np.conj(T13)*T23) / ((T22-L1)*np.conj(T13) - np.conj(T12)*np.conj(T23))

    # second eigenvector - corresponds to the eigenvalue L2
    U2 = np.ones((dims[0], dims[1], 3), 'complex64')
    U2[:, :, 0] = (L2 -T33)/np.conj(T13) + (((L2-T33)*np.conj(T12) + np.conj(T13)*T23)*np.conj(T23))/ \
                    (((T22-L2)*np.conj(T13) - np.conj(T12)*np.conj(T23))*np.conj(T13))
    U2[:, :, 1] = -((L2-T33)*np.conj(T12)+np.conj(T13)*T23) / ((T22-L2)*np.conj(T13) - np.conj(T12)*np.conj(T23))

    # third eigenvector - corresponds to the minimum eigenvalue L3
    U3 = np.ones((dims[0], dims[1], 3), 'complex64')
    U3[:, :, 0] = (L3 -T33)/np.conj(T13) + (((L3-T33)*np.conj(T12) + np.conj(T13)*T23)*np.conj(T23))/ \
                    (((T22-L3)*np.conj(T13) - np.conj(T12)*np.conj(T23))*np.conj(T13))
    U3[:, :, 1] = -((L3-T33)*np.conj(T12)+np.conj(T13)*T23) / ((T22-L3)*np.conj(T13) - np.conj(T12)*np.conj(T23))

    # normalize to get orthonormal eigenvectors
    norm1 = np.sqrt( np.abs(U1[:,:,0])**2 + np.abs(U1[:,:,1])**2 + np.abs(U1[:,:,2])**2)
    norm2 = np.sqrt( np.abs(U2[:,:,0])**2 + np.abs(U2[:,:,1])**2 + np.abs(U2[:,:,2])**2)
    norm3 = np.sqrt( np.abs(U3[:,:,0])**2 + np.abs(U3[:,:,1])**2 + np.abs(U3[:,:,2])**2)
    for nn in range(3):
        U1[:,:,nn] = U1[:,:,nn] / norm1
        U2[:,:,nn] = U2[:,:,nn] / norm2
        U3[:,:,nn] = U3[:,:,nn] / norm3

    del norm1, norm2, norm3

    return U1, U2, U3

#### Added by Marion
def calculate_eigenvalues_2(T11, T12, T22):
    #Calculate and order (from max to min) the eigenvalues of a 2x2 hermitian matrix in closed-form.
    #Inputs can be 2D az - rg (rows - columns).
    dims = T11.shape


    #Compute trace and determinant
    TR = T11 + T22 
    DET = T11 * T22 - np.abs(T12)**2
    #Compute eigenvalues using the quadratic formula
    sqrt_term = np.sqrt(TR**2 - 4 * DET) 

    LA = (TR + sqrt_term) / 2 # Largest eigenvalue 
    LB = (TR - sqrt_term) / 2 # Smallest eigenvalue

    dumm = np.zeros((dims[0], dims[1], 2), 'float32')
    dumm [:, :, 0] = np.real(LA)
    dumm [:, :, 1] = np.real(LB)

    del LA, LB

    L1 = np.max(dumm, axis = 2)
    L2 = np.min(dumm, axis = 2)

    del dumm
    

    return L1, L2



def calculate_eigenvectors_2(T11, T12, T22, L1, L2):
    # Calculate the eigenvectors corresponding to the eigenvalues (L1, L2)
    # Inputs can be 2D arrays.

    # get dimensions
    dims = T11.shape

    # first eigenvector - corresponds to the maximum eigenvalue L1
    U1 = np.zeros((dims[0], dims[1], 2), 'complex64')
    U1[:, :, 0] = (L1 - T22) / (T12 + np.conj(T11))
    U1[:, :, 1] = -T12 / (T12 + np.conj(T11))

    # second eigenvector - corresponds to the eigenvalue L2
    U2 = np.zeros((dims[0], dims[1], 2), 'complex64')
    U2[:, :, 0] = (L2 - T22) / (T12 + np.conj(T11))
    U2[:, :, 1] = 1

    # normalize to get orthonormal eigenvectors
    norm1 = np.sqrt(np.abs(U1[:,:,0])**2 + np.abs(U1[:,:,1])**2)
    norm2 = np.sqrt(np.abs(U2[:,:,0])**2 + np.abs(U2[:,:,1])**2)
    for nn in range(2):
        U1[:,:,nn] = U1[:,:,nn] / norm1
        U2[:,:,nn] = U2[:,:,nn] / norm2

    return U1, U2



import tensorflow as tf
import numpy as np
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import binom
from scipy import stats
import scipy
from sklearn.calibration import calibration_curve
from sklearn.metrics import r2_score, confusion_matrix


def kl_latent_space(z, log_det_J):
    """ Computes the Kullback-Leibler divergence (Maximum Likelihood Loss) between true and approximate
    posterior using simulated data and parameters.
    """

    loss = tf.reduce_mean(0.5 * tf.square(tf.norm(z, axis=-1)) - log_det_J)
    return loss


def maximum_mean_discrepancy(source_samples, target_samples, kernel = "gaussian", 
minimum=0., unbiased=False, squared=True):
    """ This Maximum Mean Discrepancy (MMD) loss is calculated with a number of different Gaussian or Inverse-Multiquadratic kernels.
    """

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    
    if kernel == "gaussian":
        kernel = partial(_gaussian_kernel_matrix, sigmas=sigmas)
    elif kernel == "inverse_multiquadratic":
        kernel = partial(_inverse_multiquadratic_kernel_matrix, sigmas=sigmas)
    else:
        print("Invalid kernel specified. Falling back to default Gaussian.")
        kernel = partial(_gaussian_kernel_matrix, sigmas=sigmas)
    
    if unbiased:
        loss_value = _mmd_kernel_unbiased(source_samples, target_samples, kernel=kernel)
    else:
        loss_value = _mmd_kernel(source_samples, target_samples, kernel=kernel)
        
        
    loss_value = tf.maximum(minimum, loss_value) 
    
    if squared:
        return loss_value
    else:
        return tf.math.sqrt(loss_value)


def _gaussian_kernel_matrix(x, y, sigmas):
    """ Computes a Gaussian Radial Basis Kernel between the samples of x and y.

    We create a sum of multiple gaussian kernels each having a width :math:`\sigma_i`.

    Parameters
    ----------
    x :  tf.Tensor of shape (M, num_features)
    y :  tf.Tensor of shape (N, num_features)
    sigmas : list(float)
        List which denotes the widths of each of the gaussians in the kernel.

    Returns
    -------
    kernel: tf.Tensor
        RBF kernel of shape [num_samples{x}, num_samples{y}]
    """

    norm = lambda v: tf.reduce_sum(tf.square(v), 1)
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    kernel = tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
    return kernel

def _inverse_multiquadratic_kernel_matrix(x, y, sigmas):
    """ Computes an inverse multiquadratic RBF between the samples of x and y.

    We create a sum of multiple IM-RBF kernels each having a width :math:`\sigma_i`.

    Parameters
    ----------
    x :  tf.Tensor of shape (M, num_features)
    y :  tf.Tensor of shape (N, num_features)
    sigmas : list(float)
        List which denotes the widths of each of the gaussians in the kernel.

    Returns
    -------
    kernel: tf.Tensor
        RBF kernel of shape [num_samples{x}, num_samples{y}]
    """
    
    dist = tf.expand_dims(tf.reduce_sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1), axis=-1)
    sigmas = tf.expand_dims(sigmas, 0)
    return tf.reduce_sum(sigmas / (dist + sigmas), axis=-1)

def _mmd_kernel(x, y, kernel=None):
    """ Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of the distributions of x and y.

    Parameters
    ----------
    x      : tf.Tensor of shape (num_samples, num_features)
    y      : tf.Tensor of shape (num_samples, num_features)
    kernel : callable, default: _gaussian_kernel_matrix
        A function which computes the kernel in MMD.

    Returns
    -------
    loss : tf.Tensor
        squared maximum mean discrepancy loss, shape (,)
    """

    loss = tf.reduce_mean(kernel(x, x))  
    loss += tf.reduce_mean(kernel(y, y))  
    loss -= 2 * tf.reduce_mean(kernel(x, y))
    return loss

def mmd_kl_loss(network, *args, mmd_weight=1.0, kernel="gaussian"):
    """KL loss in latent z space, MMD loss in summary space."""
    
    # Apply net and unpack 
    x_sum, out = network(*args, return_summary=True)
    z, log_det_J = out
    
    # Apply MMD loss to x_sum
    z_normal = tf.random.normal(x_sum.shape) # idea: alpha-stable, alpha=1.5
    mmd_loss = maximum_mean_discrepancy(x_sum, z_normal, kernel=kernel)
    
    # Apply KL loss for inference net
    kl_loss = kl_latent_space(z, log_det_J)
    
    # Sum and return losses
    return kl_loss + mmd_weight * mmd_loss



def _mmd_kernel_unbiased(x, y, kernel=None):
    """ Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of the distributions of x and y.

    Parameters
    ----------
    x      : tf.Tensor of shape (num_samples, num_features)
    y      : tf.Tensor of shape (num_samples, num_features)
    kernel : callable, default: _gaussian_kernel_matrix
        A function which computes the kernel in MMD.

    Returns
    -------
    loss : tf.Tensor
        squared maximum mean discrepancy loss, shape (,)
    """
    m, n = x.shape[0], y.shape[0]
    loss = (1.0/(m*(m-1))) * tf.reduce_sum(kernel(x, x))  
    loss += (1.0/(n*(n-1))) * tf.reduce_sum(kernel(y, y)) 
    loss -= (2.0/(m*n)) * tf.reduce_sum(kernel(x, y)) 
    return loss




####

def mahalanobis_distance_2D_1D(data, ref, cov):
    n = data.shape[0]
    mahalanobis_distances = [scipy.spatial.distance.mahalanobis(data[i], ref, cov) for i in range(n)]
    return np.array(mahalanobis_distances)



####

def true_vs_estimated(theta_true, theta_est, param_names, dpi=300,
                      figsize=(20, 4), show=True, filename=None, font_size=12):
    """ Plots a scatter plot with abline of the estimated posterior means vs true values.
    Parameters
    ----------
    theta_true: np.array
        Array of true parameters.
    theta_est: np.array
        Array of estimated parameters.
    param_names: list(str)
        List of parameter names for plotting.
    dpi: int, default:300
        Dots per inch (dpi) for the plot.
    figsize: tuple(int, int), default: (20,4)
        Figure size.
    show: boolean, default: True
        Controls if the plot will be shown
    filename: str, default: None
        Filename if plot shall be saved
    font_size: int, default: 12
        Font size
    """


    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Determine n_subplots dynamically
    n_row = int(np.ceil(len(param_names) / 6))
    n_col = int(np.ceil(len(param_names) / n_row))

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat
        
    # --- Plot true vs estimated posterior means on a single row --- #
    for j in range(len(param_names)):
        
        # Plot analytic vs estimated
        axarr[j].scatter(theta_est[:, j], theta_true[:, j], color='black', alpha=0.4)
        
        # get axis limits and set equal x and y limits
        lower_lim = min(axarr[j].get_xlim()[0], axarr[j].get_ylim()[0])
        upper_lim = max(axarr[j].get_xlim()[1], axarr[j].get_ylim()[1])
        axarr[j].set_xlim((lower_lim, upper_lim))
        axarr[j].set_ylim((lower_lim, upper_lim))
        axarr[j].plot(axarr[j].get_xlim(), axarr[j].get_xlim(), '--', color='black')
        
        # Compute NRMSE
        rmse = np.sqrt(np.mean( (theta_est[:, j] - theta_true[:, j])**2 ))
        nrmse = rmse / (theta_true[:, j].max() - theta_true[:, j].min())
        axarr[j].text(0.1, 0.9, 'NRMSE={:.3f}'.format(nrmse),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes,
                     size=10)
        
        # Compute R2
        #r2 = r2_score(theta_true[:, j], theta_est[:, j])
        #axarr[j].text(0.1, 0.8, '$R^2$={:.3f}'.format(r2),
        #             horizontalalignment='left',
        #             verticalalignment='center',
        #             transform=axarr[j].transAxes, 
        #             size=10)
        
        if j == 0:
            # Label plot
            axarr[j].set_xlabel('Estimated')
            axarr[j].set_ylabel('True')
        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)
    
    # Adjust spaces
    f.tight_layout()

    if filename is not None:
        f.savefig(filename, bbox_inches="tight")
        
        
    if show:
        plt.show()
        
        
def mmd_permutation(x, y, n_perm=1000, kernel=_gaussian_kernel_matrix):
    """
    Computes the p-value of the MMD by permuting the samples.
    """
    
    # Obtain sample sizes
    m = int(x.shape[0])
    n = int(y.shape[0])
    xy = tf.concat((x, y), axis=0)
    
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(_gaussian_kernel_matrix, sigmas=sigmas)
    
    mmd2_null = np.zeros(n_perm)
    # Run permutations
    for i in range(n_perm):
        idx = np.random.permutation(m+n)
        xy = tf.gather(xy, idx, axis=0)
        mmd2 = _mmd_kernel_unbiased(xy[:m], xy[m:], kernel=gaussian_kernel)
        mmd2_null[i] = mmd2
    return mmd2_null



def normalize(x):
    n_sim, n_obs, data_dim = x.shape
    s = np.std(x, axis=1)
    s_reshaped = s.reshape(n_sim, 1, data_dim).repeat(n_obs, axis=1)
    x_normalized = np.divide(x, s_reshaped)
    return x_normalized



'''
def noisify_x(x, lamda, alpha=1.0, noise_scale=1.0):
    n_sim, n_obs, data_dim = x.shape
    
    s_x = np.std(x, axis=1).reshape(n_sim, 1, data_dim).repeat(n_obs, axis=1)
    
    xi = scipy.stats.levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=noise_scale, size=x.shape)
    x_o = x + lamda * xi
    
    s_x_o = np.std(x_o, axis=1).reshape(n_sim, 1, data_dim).repeat(n_obs, axis=1)
    
    normalization_factors = np.divide(s_x_o, s_x)
    
    x_o_normalized = np.divide(x_o, normalization_factors)
    
    return x_o_normalized
    '''

'''
def noisify_x(x, lamda, alpha=1.5, noise_scale=1.0):
    n_sim, n_obs, data_dim = x.shape
    N_replacements = int(lamda * n_obs)

    indices = [np.random.choice(n_obs, size = N_replacements, replace=False) for i in range(n_sim)]

    xi = scipy.stats.levy_stable.rvs(
        alpha=alpha, beta=0, loc=0, scale=noise_scale, size=(n_sim, N_replacements, data_dim)
    )

    for i, idx in enumerate(indices):
        x[i, idx, :] = xi[i]
    
    return x
    '''

def noisify_x(x, lamda, noise_shape=1.0, noise_sampler=partial(scipy.stats.levy_stable.rvs, alpha=1.5, beta=0, loc=0, scale=1)):
    n_sim, n_obs, data_dim = x.shape
    N_replacements = int(lamda * n_obs)

    indices = [np.random.choice(n_obs, size = N_replacements, replace=False) for i in range(n_sim)]

    xi = noise_sampler(size=(n_sim, N_replacements, data_dim))

    for i, idx in enumerate(indices):
        x[i, idx, :] = xi[i]
    
    return x



def beta_noise_sampler(a, b, tau, size, mu=0):
    x = scipy.stats.beta.rvs(a=a, b=b, size=size)
    x = x - a/(a+b) + mu
    x = x*3*tau
    return x




def build_viridis_palette(n, n_total=20, base_palette="viridis"):
    """
    Builds a viridis palette with maximal entropy (evenly spaced)
    """
    color_palette = np.array(sns.color_palette(base_palette, n_colors=n_total))
    indices = np.array(np.floor(np.linspace(0, n_total-1, n)), dtype=np.int32)
    color_palette = color_palette[indices]
    return [tuple(c) for c in color_palette]
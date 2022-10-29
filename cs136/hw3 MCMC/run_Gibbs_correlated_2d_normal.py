'''
Usage
-----
Run as a script with no arguments
```
$ python run_Gibbs_correlated_2d_normal.py
```
Will run MCMC and produce plots

Purpose
-------
Sample from a 2-dim. Normal distribution using Gibbs Sampling

Target distribution:
# mean
>>> mu_D = np.asarray([-1.0, 1.0])
# covariance
>>> cov_DD = np.asarray([[2.0, 0.95], [0.95, 1.0]])

'''

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from GibbsSampler import GibbsSampler2D

def draw_z0_given_z1(z1, u_stdnorm):
    ''' Draw sample from target conditional for z0 given z1

    Does not generate any randomness internally.
    Relies performing a transformation,
    Given a passed value "u" drawn externally from standard normal.

    Args
    ----
    z1 : scalar float
        Value of variable z1 to condition on.
    u_stdnorm : scalar float
        Value drawn from Standard Normal distribution
        Assumed: u ~ Normal(0.0, 1.0)
        Should be only source of randomness you need.

    Returns
    -------
    z0 : scalar float
        Sample from conditional p*(z0 | z1)
    '''
    # TODO Compute conditional distribution mean and covariance
    mu_D = np.asarray([-1.0, 1.0])
    cov_DD = np.asarray([[2.0, 0.95], [0.95, 1.0]])
    a = 0
    b = 1
    var_01 = cov_DD[a, a] - (cov_DD[a, b] * cov_DD[b, a]) / cov_DD[b, b]
    mean_01 = mu_D[a] + cov_DD[a, b] / cov_DD[b, b] * (z1 - mu_D[b])
    # Given known joint Gaussian over z0, z1 with mean mu_D and covar cov_DD

    # TODO Transform provided u_stdnorm value into sample from target conditional
    z0_samp = mean_01 + np.sqrt(var_01) * u_stdnorm # fixme
    return z0_samp

def draw_z1_given_z0(z0, u_stdnorm):
    ''' Draw sample from target conditional for z1 given z0

    Does not generate any randomness internally.
    Relies performing a transformation,
    Given a passed value "u" drawn externally from standard normal.

    Args
    ----
    z1 : scalar float
        Value of variable z0 to condition on.
    u_stdnorm : value drawn from Standard Normal distribution
        Assumed: u ~ Normal(0.0, 1.0)
        Should be only source of randomness you need.

    Returns
    -------
    z1 : scalar float
        Sample from conditional p*(z1 | z0)
    '''

    # Use Bishop PRML Equations 2.81 and 2.82 to compute conditional,
    # Given joint Gaussian over z0, z1 with mean mu_D and covar cov_DD
    mu_D = np.asarray([-1.0, 1.0])
    cov_DD = np.asarray([[2.0, 0.95], [0.95, 1.0]])
    a = 1
    b = 0
    var_10 = cov_DD[a,a] - (cov_DD[a,b] * cov_DD[b,a]) / cov_DD[b,b]
    mean_10 = mu_D[a] + cov_DD[a,b] / cov_DD[b,b] * (z0 - mu_D[b])

    # Draw from p(z1 | z0) =  Normal(mean_10, var_10)
    z1_samp = mean_10 + np.sqrt(var_10) * u_stdnorm
    return z1_samp


if __name__ == '__main__':
    ''' Main block to test Gibbs Sampler for D=2-dim. random. variable z_D
   
    Goal: Run separate MCMC chains from two initial values and verify that
    the sampler *converges* to the same distribution in both chains
    '''
    n_samples = 10000   # total number of iterations of MCMC
    n_keep = 5000       # number samples to keep
    random_state = 42   # seed for random number generator

    # Two initializations, labeled 'A' and 'B'
    z_initA_D = np.zeros(2)
    z_initB_D = np.asarray([1.0, -1.0])

    # No hyperparameters to tune for Gibbs
    G = 1

    # Prepare a plot to view samples from two chains (A/B) side-by-side
    _, ax_grid = plt.subplots(
        nrows=2, ncols=1, sharex=True, sharey=True,
        figsize=(2*G, 2*2))

    # Create samplers and run them for specified num iterations
    samplerA = GibbsSampler2D(draw_z0_given_z1, draw_z1_given_z0, random_state)
    z_fromA_list, samplerA_info = samplerA.draw_samples(zinit_D=z_initA_D, n_samples=n_samples)

    samplerB = GibbsSampler2D(draw_z0_given_z1, draw_z1_given_z0, random_state+1)
    z_fromB_list, samplerB_info = samplerB.draw_samples(zinit_D=z_initB_D, n_samples=n_samples)

    # Stack list of samples into a 2D array of size (S, D)
    # Keeping only the last few samples (and thus discarding burnin)
    zA_SD = np.vstack(z_fromA_list[-n_keep:])
    zB_SD = np.vstack(z_fromB_list[-n_keep:])

    # Plot samples as scatterplot
    # Use small alpha transparency value for visual debugging of rare/frequent samples
    ax_grid[0].plot(zA_SD[:,0], zA_SD[:,1], 'r.', alpha=0.05)
    ax_grid[1].plot(zB_SD[:,0], zB_SD[:,1], 'b.', alpha=0.05)
    # Mark initial points with "X"
    ax_grid[0].plot(z_fromA_list[0][0], z_fromA_list[0][1], 'rx')
    ax_grid[1].plot(z_fromB_list[0][0], z_fromB_list[0][1], 'bx')

    ##Label axes
    ax_grid[0].set_xlabel("z_0")
    ax_grid[1].set_xlabel("z_0")
    ax_grid[0].set_ylabel("z_1")
    ax_grid[1].set_ylabel("z_1")

    ##Title for plots
    ax_grid[0].set_title("Initialization A")
    ax_grid[1].set_title("Initialization B")

    # Pretty print some stats for the samples
    # To give a way to check "convergence" from the terminal's stdout
    msg_pattern = ("Gibbs from init %s | kept %d of %d samples | accept rate %.3f"
        + "\n    percentiles z0: 10th % 5.2f   50th % 5.2f   90th % 5.2f" 
        + "\n    percentiles z1: 10th % 5.2f   50th % 5.2f   90th % 5.2f"
        )
    print(msg_pattern % (
        'A', n_keep, n_samples,
        samplerA_info['accept_rate_last_half'],
        *tuple(np.percentile(zA_SD[:,0:1], [10, 50, 90], axis=0)),
        *tuple(np.percentile(zA_SD[:,1:2], [10, 50, 90], axis=0)),
        ))
    print(msg_pattern % (
        'B', n_keep, n_samples,
        samplerB_info['accept_rate_last_half'],
        *tuple(np.percentile(zB_SD[:,0:1], [10, 50, 90], axis=0)),
        *tuple(np.percentile(zB_SD[:,1:2], [10, 50, 90], axis=0)),
        ))

    # Make plots pretty and standardized
    for ax in ax_grid.flatten():
        ax.set_xlim([-5, 5]);
        ax.set_ylim([-5, 5]);
        ax.set_aspect('equal', 'box');
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
    plt.tight_layout()
    plt.savefig("problem2_figure.pdf", bbox_to_inches='tight', pad_inches=0)
    plt.show()
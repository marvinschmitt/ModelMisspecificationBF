# from https://github.com/danielward27/rnpe/blob/main/rnpe/tasks.py



from abc import ABC, abstractmethod
import os
import jax.numpy as jnp
import numpy as onp
from jax import random
import jax
import numba


class Task(ABC):
    @property
    def name(self):
        return type(self).__name__

    @abstractmethod
    def sample_prior(self, key: random.PRNGKey, n: int):
        "Draw n samples from the prior."
        pass

    @abstractmethod
    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray):
        "Carry out simulations."
        pass

    @abstractmethod
    def generate_observation(self, key: random.PRNGKey, misspecified=True):
        "Generate misspecified pseudo-observed data. Returns (theta_true, y)."
        pass

    def generate_dataset(
        self, key: random.PRNGKey, n: int, scale=True, misspecified=True,
    ):
        "Generate optionally scaled dataset with pseudo-observed value and simulations"
        theta_key, x_key, obs_key = random.split(key, 3)
        theta = self.sample_prior(theta_key, n)
        x = self.simulate(x_key, theta)
        x = remove_nans_and_warn(x)
        theta_true, y, y_raw = self.generate_observation(
            obs_key, misspecified=misspecified
        )

        if scale:
            theta, theta_true, x, y = self.scale(theta, theta_true, x, y)

        data = {
            "theta": theta,
            "theta_true": theta_true,
            "x": x,
            "y": y,
            "y_raw": y_raw,
        }

        return data

    def scale(self, theta, theta_true, x, y):
        "Center and scale data, using mean and std from theta and x."
        theta_means, theta_stds = theta.mean(axis=0), theta.std(axis=0)
        theta = (theta - theta_means) / theta_stds
        theta_true = (theta_true - theta_means) / theta_stds

        x_means, x_stds = x.mean(axis=0), x.std(axis=0)
        x = (x - x_means) / x_stds
        y = (y - x_means) / x_stds

        self.scales = {
            "theta_mean": theta_means,
            "theta_std": theta_stds,
            "x_mean": x_means,
            "x_std": x_stds,
        }
        return theta, theta_true, x, y

    def in_prior_support(self, theta):
        return jnp.full(theta.shape[0], True)

    def get_true_posterior_samples(self, key: random.PRNGKey, y: jnp.ndarray, n: int):
        raise NotImplementedError(
            "This task does not have a method for sampling the true posterior implemented."
        )


class CS(Task):
    def __init__(self):
        self.theta_names = [r"$\lambda_c$", r"$\lambda_p$", r"$\lambda_d$"]
        self.x_names = [
            "N Stromal",
            "N Cancer",
            "Mean Min Dist",
            "Max Min Dist",
        ]
        #self.cell_rate_lims = {"minval": 200, "maxval": 1500}
        #self.parent_rate_lims = {"minval": 3, "maxval": 20}
        #self.daughter_rate_lims = {"minval": 10, "maxval": 20}
        
        self.cell_rate_prior_params = {"shape": 25, "scale": 1/0.03}
        self.parent_rate_prior_params = {"shape": 5, "scale": 1/0.5}
        self.daughter_rate_prior_params = {"shape": 45, "scale": 1/3}
        
        self.tractable_posterior = False

    def simulate(
        self,
        key: random.PRNGKey,
        theta: jnp.array,
        summarise: bool = True,
        necrosis: bool = False,
        pi_necrosis = 0.75
    ):
        theta = onp.array(theta)
        onp.random.seed(key[1].item())

        x = []
        for row in theta:
            xi = self._simulate(
                cell_rate=row[0],
                cancer_parent_rate=row[1],
                cancer_daughter_rate=row[2],
                necrosis=necrosis,
                pi_necrosis = pi_necrosis
            )
            if summarise is True:
                xi = self.summarise(*xi)
            x.append(xi)

        if summarise:
            x = jnp.row_stack(x)

        return x

    def _simulate(
        self,
        cell_rate: float = 1000,
        cancer_parent_rate: float = 5,
        cancer_daughter_rate: float = 30,
        necrosis=False,
        pi_necrosis = 0.75
    ):
        num_cells = onp.random.poisson(cell_rate)
        cells = onp.random.uniform(size=(num_cells, 2))

        num_cancer_parents = onp.random.poisson(cancer_parent_rate) + 1
        cancer_parents = onp.random.uniform(0, 1, size=(num_cancer_parents, 2))

        num_cancer_daughters = (
            onp.random.poisson(cancer_daughter_rate, (num_cancer_parents,)) + 1
        )

        dists = dists_between(cells, cancer_parents)  # n_cells by n_parents
        radii = self.n_nearest_dists(dists, num_cancer_daughters)
        is_cancer = (dists <= radii).any(axis=1)

        if necrosis:
            has_necrosis = onp.random.binomial(1, p=pi_necrosis, size=(num_cancer_parents,))
            has_necrosis = has_necrosis.astype(bool)
            if has_necrosis.sum() > 0:
                bl_array = dists[:, has_necrosis] < (radii[has_necrosis] * 0.8)
                necrotized = onp.any(bl_array, axis=1)
                cells, is_cancer = (
                    cells[~necrotized],
                    is_cancer[~necrotized],
                )

        return cells, is_cancer

    def summarise(self, cells, is_cancer, threshold_n_stromal: int = 50):
        """Calculate summary statistics, threshold_n_stromal limits the number
        of stromal cells (trade off for efficiency)."""
        num_cancer = is_cancer.sum()

        if num_cancer == is_cancer.shape[0]:
            print("Warning, no stromal cells. Returning nan for summary statistics.")
            return jnp.full(len(self.x_names), jnp.nan)

        num_stromal = (~is_cancer).sum()
        threshold_num_stromal = min(threshold_n_stromal, num_stromal)
        cancer = cells[is_cancer]
        stromal = cells[~is_cancer][:threshold_num_stromal]

        dists = dists_between(stromal, cancer)

        min_dists = dists.min(axis=1)
        mean_nearest_cancer = min_dists.mean()
        max_nearest_cancer = min_dists.max()

        summaries = [
            num_stromal,
            num_cancer,
            mean_nearest_cancer,
            max_nearest_cancer,
        ]

        return jnp.array(summaries)

    def generate_observation(self, key: random.PRNGKey, misspecified=True):
        theta_key, y_key = random.split(key)
        theta_true = self.sample_prior(theta_key, 1)
        y_raw = self.simulate(y_key, theta_true, necrosis=misspecified, summarise=False)
        y = self.summarise(*y_raw[0])
        return jnp.squeeze(theta_true), jnp.squeeze(y), y_raw

    def sample_prior(self, key: random.PRNGKey, n: int):
        keys = random.split(key, 3)
        #cell_rate = random.uniform(keys[0], (n,), **self.cell_rate_lims)
        #cancer_parent_rate = random.uniform(keys[1], (n,), **self.parent_rate_lims)
        #cancer_daughter_rate = random.uniform(keys[2], (n,), **self.daughter_rate_lims,)
        cell_rate = jnp.asarray(onp.random.gamma(**self.cell_rate_prior_params, size=(n,)))
        cancer_parent_rate = jnp.asarray(onp.random.gamma(**self.parent_rate_prior_params, size=(n,)))
        cancer_daughter_rate = jnp.asarray(onp.random.gamma(**self.daughter_rate_prior_params, size=(n,)))
        
        return jnp.column_stack([cell_rate, cancer_parent_rate, cancer_daughter_rate])

    def n_nearest_dists(self, dists, n_points):
        "Minimum distance containing n points. n_points to match axis dists in axis 1."
        assert dists.shape[1] == len(n_points)
        d_sorted = onp.partition(dists, kth=n_points, axis=0)
        min_d = d_sorted[n_points, onp.arange(dists.shape[1])]
        return min_d

    def in_prior_support(self, theta):
        bools = []
        limits = [self.cell_rate_lims, self.parent_rate_lims, self.daughter_rate_lims]
        for col, lims in zip(theta.T, limits):
            in_support = (col > lims["minval"]) & (col < lims["maxval"])
            bools.append(in_support)
        return jnp.column_stack(bools).all(axis=1)


@numba.njit(fastmath=True)
def dists_between(a, b):
    "Returns a.shape[0] by b.shape[0] l2 norms between rows of arrays."
    dists = []
    for ai in a:
        for bi in b:
            dists.append(onp.linalg.norm(ai - bi))
    return onp.array(dists).reshape(a.shape[0], b.shape[0])


def remove_nans_and_warn(x):
    nan_rows = jnp.any(jnp.isnan(x), axis=1)
    n_nan = nan_rows.sum()
    if n_nan > 0:
        x = x[~nan_rows]
        print(f"Warning {n_nan} simulations contained NAN values have been removed.")
    return x
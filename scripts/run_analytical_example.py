"""Run a clean analytical benchmark comparing full-space and PCA-reduced optimization.

This script is designed to perform the following:
1. Builds a synthetic benchmark problem.
2. Generates an ensemble of designs.
3. Fits PCA to the ensemble.
4. Runs full-space and reduced-space optimization.
5. Saves summary files and comparison plots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA


@dataclass
class AnalyticalConfig:
    benchmark_name: str = "latent_quadratic"
    full_dim: int = 60
    latent_dim: int = 10
    pca_components: int = 10
    ensemble_size: int = 300
    initial_guess_index: int = 0
    dynamic_scaling: bool = False
    whitening: bool = False
    full_method: str = "L-BFGS-B"
    reduced_method: str = "trust-constr"
    maxiter_full: int = 400
    maxiter_reduced: int = 400
    random_seed: int = 10
    output_dir: Path = Path("results/analytical")
    plot_results: bool = True
    figure_size: tuple[float, float] = (8, 6)


@dataclass
class BenchmarkProblem:
    name: str
    dim: int
    objective: Callable[[np.ndarray], float]
    gradient: Callable[[np.ndarray], np.ndarray]
    hessian: Callable[[np.ndarray], np.ndarray]
    minimizer: Optional[np.ndarray] = None
    latent_dim: Optional[int] = None
    latent_basis: Optional[np.ndarray] = None
    latent_mean: Optional[np.ndarray] = None
    latent_target: Optional[np.ndarray] = None


class FullSpaceTracker:
    def __init__(self, problem: BenchmarkProblem) -> None:
        self.problem = problem
        self.f_history: list[float] = []
        self.grad_norm_history: list[float] = []
        self.x_history: list[np.ndarray] = []

    def objective_and_gradient(self, x: np.ndarray):
        f = self.problem.objective(x)
        g = self.problem.gradient(x)
        self.f_history.append(float(f))
        self.grad_norm_history.append(float(np.linalg.norm(g)))
        self.x_history.append(x.copy())
        return f, g


class ReducedPCAOptimizer:
    def __init__(self, problem: BenchmarkProblem, pca: PCA, dynamic_scaling: bool, whitening: bool) -> None:
        self.problem = problem
        self.pca = pca
        self.dynamic_scaling = dynamic_scaling
        self.whitening = whitening
        self.scale_factors = np.sqrt(np.maximum(pca.explained_variance_, 1e-12)) if whitening else np.ones(pca.n_components_)
        self.scaling_factor: Optional[float] = None
        self.raw_f_history: list[float] = []
        self.grad_norm_history: list[float] = []
        self.z_history: list[np.ndarray] = []

    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        coeffs = z * self.scale_factors
        return self.pca.inverse_transform(coeffs.reshape(1, -1)).reshape(-1)

    def project_gradient(self, full_grad: np.ndarray) -> np.ndarray:
        grad_t = full_grad @ self.pca.components_.T
        return grad_t * self.scale_factors

    def reduced_hessian(self, z: np.ndarray) -> np.ndarray:
        x = self.reconstruct(z)
        h_full = self.problem.hessian(x)
        basis = self.pca.components_.T
        h_t = basis.T @ h_full @ basis
        h_z = h_t * np.outer(self.scale_factors, self.scale_factors)
        if self.scaling_factor is not None:
            h_z = self.scaling_factor * h_z
        return h_z

    def objective_and_gradient(self, z: np.ndarray):
        x = self.reconstruct(z)
        f_raw = self.problem.objective(x)
        g_full = self.problem.gradient(x)
        g_red = self.project_gradient(g_full)
        grad_max = max(np.max(np.abs(g_red)), 1e-15)
        if self.scaling_factor is None:
            self.scaling_factor = 1.0 / grad_max if self.dynamic_scaling else 1.0
        f_scaled = f_raw * self.scaling_factor
        g_scaled = g_red * self.scaling_factor
        self.raw_f_history.append(float(f_raw))
        self.grad_norm_history.append(float(np.linalg.norm(g_scaled)))
        self.z_history.append(z.copy())
        return f_scaled, g_scaled


def latent_quadratic_problem(dim: int, latent_dim: int, seed: int) -> BenchmarkProblem:
    rng = np.random.default_rng(seed)
    mean = rng.normal(0.0, 0.5, size=dim)
    basis = rng.normal(size=(dim, latent_dim))
    basis, _ = np.linalg.qr(basis)
    basis = basis[:, :latent_dim]
    q = rng.normal(size=(latent_dim, latent_dim))
    q = q.T @ q + 0.5 * np.eye(latent_dim)
    b = rng.normal(size=latent_dim)
    z_star = np.linalg.solve(q, b)
    x_star = mean + basis @ z_star

    def latent_coords(x: np.ndarray) -> np.ndarray:
        return basis.T @ (x - mean)

    def objective(x: np.ndarray) -> float:
        d = latent_coords(x) - z_star
        return float(0.5 * d @ q @ d)

    def gradient(x: np.ndarray) -> np.ndarray:
        d = latent_coords(x) - z_star
        return basis @ (q @ d)

    def hessian(_: np.ndarray) -> np.ndarray:
        return basis @ q @ basis.T

    return BenchmarkProblem(
        name="latent_quadratic",
        dim=dim,
        objective=objective,
        gradient=gradient,
        hessian=hessian,
        minimizer=x_star,
        latent_dim=latent_dim,
        latent_basis=basis,
        latent_mean=mean,
        latent_target=z_star,
    )


def build_problem(config: AnalyticalConfig) -> BenchmarkProblem:
    if config.benchmark_name != "latent_quadratic":
        raise ValueError("This clean public version currently supports only 'latent_quadratic'.")
    return latent_quadratic_problem(config.full_dim, config.latent_dim, config.random_seed)


def build_ensemble(problem: BenchmarkProblem, ensemble_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(ensemble_size, problem.latent_dim)) * 2.0
    noise = 0.02 * rng.normal(size=(ensemble_size, problem.dim))
    return problem.latent_mean + z @ problem.latent_basis.T + noise



def save_history_plot(values_a, values_b, labels, ylabel, path: Path, logy: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(values_a, ls='-', marker='o', ms=3, label=labels[0])
    ax.plot(values_b, ls='--', marker='s', ms=3, label=labels[1])
    ax.set_xlabel('Objective evaluations')
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3, ls=':')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)



def save_latent_landscape(problem: BenchmarkProblem, tracker: ReducedPCAOptimizer, initial_z: np.ndarray, final_z: np.ndarray, outdir: Path) -> None:
    if problem.latent_target is None or len(initial_z) < 2:
        return
    z0 = initial_z.copy()
    z_star = problem.latent_target
    grid_n = 180
    z1_min = min(z0[0], final_z[0], z_star[0]) - 2.0
    z1_max = max(z0[0], final_z[0], z_star[0]) + 2.0
    z2_min = min(z0[1], final_z[1], z_star[1]) - 2.0
    z2_max = max(z0[1], final_z[1], z_star[1]) + 2.0
    z1 = np.linspace(z1_min, z1_max, grid_n)
    z2 = np.linspace(z2_min, z2_max, grid_n)
    Z1, Z2 = np.meshgrid(z1, z2)
    F = np.zeros_like(Z1)
    z_fixed = z0.copy()
    for i in range(grid_n):
        for j in range(grid_n):
            z = z_fixed.copy()
            z[0] = Z1[i, j]
            z[1] = Z2[i, j]
            F[i, j] = problem.objective(tracker.reconstruct(z))
    path = np.array(tracker.z_history)
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(Z1, Z2, np.log10(F + 1e-20), levels=50, cmap='viridis')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(r'$\log_{10}[F(\mathbf{z})]$')
    if len(path) > 0:
        ax.plot(path[:, 0], path[:, 1], '-o', color='white', lw=1.2, ms=3, label='Path')
        ax.scatter(path[0, 0], path[0, 1], color='red', s=30, label='Start', zorder=3)
        ax.scatter(path[-1, 0], path[-1, 1], color='cyan', s=30, label='Final', zorder=3)
    ax.scatter(z_star[0], z_star[1], color='black', s=50, marker='*', label='Optimum', zorder=3)
    ax.set_xlabel('Latent coordinate z1')
    ax.set_ylabel('Latent coordinate z2')
    ax.legend(loc='best', framealpha=0.8)
    fig.tight_layout()
    fig.savefig(outdir / 'latent_landscape_path.png', dpi=600)
    plt.close(fig)


def run(config: AnalyticalConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    problem = build_problem(config)
    ensemble = build_ensemble(problem, config.ensemble_size, config.random_seed)
    pca = PCA(n_components=config.pca_components)
    all_coefficients = pca.fit_transform(ensemble)

    initial_full = ensemble[config.initial_guess_index].copy()
    initial_coeffs_raw = all_coefficients[config.initial_guess_index].copy()

    full_tracker = FullSpaceTracker(problem)
    reduced_tracker = ReducedPCAOptimizer(problem, pca, config.dynamic_scaling, config.whitening)
    initial_z = initial_coeffs_raw / reduced_tracker.scale_factors

    full_result = minimize(
        fun=lambda x: full_tracker.objective_and_gradient(x),
        x0=initial_full,
        jac=True,
        method=config.full_method,
        options={'maxiter': config.maxiter_full, 'disp': True},
    )

    reduced_result = minimize(
        fun=lambda z: reduced_tracker.objective_and_gradient(z),
        x0=initial_z,
        jac=True,
        hess=lambda z: reduced_tracker.reduced_hessian(z),
        method=config.reduced_method,
        options={'maxiter': config.maxiter_reduced, 'verbose': 2},
    )

    reduced_x_star = reduced_tracker.reconstruct(reduced_result.x)
    full_final_obj = float(problem.objective(full_result.x))
    full_final_grad = float(np.linalg.norm(problem.gradient(full_result.x)))
    reduced_final_obj = float(problem.objective(reduced_x_star))
    reduced_final_grad = float(np.linalg.norm(problem.gradient(reduced_x_star)))

    full_iters = len(full_tracker.f_history)
    reduced_iters = len(reduced_tracker.raw_f_history)
    summary = {
        'benchmark': problem.name,
        'full_dim': problem.dim,
        'latent_dim': config.latent_dim,
        'pca_components': config.pca_components,
        'ensemble_size': config.ensemble_size,
        'explained_variance_ratio': float(np.sum(pca.explained_variance_ratio_)),
        'full_space': {
            'method': config.full_method,
            'success': bool(full_result.success),
            'iterations': int(full_iters),
            'final_objective': full_final_obj,
            'final_grad_norm': full_final_grad,
        },
        'reduced_space': {
            'method': config.reduced_method,
            'success': bool(reduced_result.success),
            'iterations': int(reduced_iters),
            'final_objective': reduced_final_obj,
            'final_grad_norm': reduced_final_grad,
            'dynamic_scaling': bool(config.dynamic_scaling),
            'whitening_enabled': bool(config.whitening),
        },
        'comparison': {
            'iteration_reduction_percent': 100.0 * (full_iters - reduced_iters) / max(full_iters, 1),
            'objective_improvement_factor': full_final_obj / max(reduced_final_obj, 1e-300),
            'objective_reduction_percent': 100.0 * (full_final_obj - reduced_final_obj) / max(full_final_obj, 1e-300),
        },
    }

    (config.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.save(config.output_dir / 'full_solution.npy', full_result.x)
    np.save(config.output_dir / 'reduced_coefficients_opt.npy', reduced_result.x)
    np.save(config.output_dir / 'reduced_solution_full_space.npy', reduced_x_star)

    save_history_plot(full_tracker.f_history, reduced_tracker.raw_f_history, ('Full space', 'Latent space'), 'Objective function value', config.output_dir / 'objective_history.png')
    save_history_plot(full_tracker.grad_norm_history, reduced_tracker.grad_norm_history, ('Full space', 'Latent space'), 'Gradient norm', config.output_dir / 'gradient_norm_history.png')
    if config.plot_results:
        save_latent_landscape(problem, reduced_tracker, initial_z, reduced_result.x, config.output_dir)
    return summary


if __name__ == '__main__':
    cfg = AnalyticalConfig()
    result = run(cfg)
    print(json.dumps(result, indent=2))

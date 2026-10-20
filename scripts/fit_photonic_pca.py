from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PCAConfig:
    input_data_path: Path = Path("data/raw/ensemble_data.npy")
    model_output_path: Path = Path("data/models/photonic_pca.joblib")
    coefficients_output_path: Path = Path("data/models/photonic_coefficients.npy")
    explained_variance_output_path: Path = Path(
        "data/models/explained_variance_ratio.npy"
    )
    cumulative_variance_output_path: Path = Path(
        "data/models/cumulative_explained_variance.npy"
    )
    reconstruction_output_path: Path = Path(
        "data/models/reconstructed_ensemble.npy"
    )
    output_figure_dir: Path = Path("results/photonics/pca")
    n_components: int = 8
    preview_device_index: int = 1
    figure_dpi: int = 300


def ensure_output_directories(config: PCAConfig) -> None:
    config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.coefficients_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.explained_variance_output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    config.cumulative_variance_output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    config.reconstruction_output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    config.output_figure_dir.mkdir(parents=True, exist_ok=True)


def load_ensemble(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Ensemble dataset not found: {path}\n"
            "Run scripts/generate_photonic_devices.py first."
        )

    ensemble = np.asarray(np.load(path), dtype=float)

    if ensemble.ndim != 3:
        raise ValueError(
            "Expected ensemble data with shape "
            "(n_devices, n_x, n_y), "
            f"but received shape {ensemble.shape}."
        )

    if not np.all(np.isfinite(ensemble)):
        raise ValueError("The ensemble contains NaN or infinite values.")

    return ensemble


def validate_configuration(
    ensemble: np.ndarray,
    config: PCAConfig,
) -> None:
    n_devices, n_x, n_y = ensemble.shape
    max_components = min(n_devices, n_x * n_y)

    if not 1 <= config.n_components <= max_components:
        raise ValueError(
            f"n_components must be between 1 and {max_components}; "
            f"received {config.n_components}."
        )

    if not 0 <= config.preview_device_index < n_devices:
        raise IndexError(
            f"preview_device_index must be between 0 and {n_devices - 1}; "
            f"received {config.preview_device_index}."
        )


def fit_pca(
    ensemble: np.ndarray,
    n_components: int,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    n_devices, n_x, n_y = ensemble.shape
    flattened_ensemble = ensemble.reshape(n_devices, n_x * n_y)

    pca = PCA(n_components=n_components)
    coefficients = pca.fit_transform(flattened_ensemble)
    reconstructed_flattened = pca.inverse_transform(coefficients)
    reconstructed_ensemble = reconstructed_flattened.reshape(
        n_devices, n_x, n_y
    )

    return pca, coefficients, reconstructed_ensemble


def save_model_outputs(
    pca: PCA,
    coefficients: np.ndarray,
    reconstructed_ensemble: np.ndarray,
    config: PCAConfig,
) -> None:
    joblib.dump(pca, config.model_output_path)
    np.save(config.coefficients_output_path, coefficients)
    np.save(
        config.explained_variance_output_path,
        pca.explained_variance_ratio_,
    )
    np.save(
        config.cumulative_variance_output_path,
        np.cumsum(pca.explained_variance_ratio_),
    )
    np.save(config.reconstruction_output_path, reconstructed_ensemble)


def plot_explained_variance(
    pca: PCA,
    output_path: Path,
    dpi: int,
) -> None:
    component_numbers = np.arange(1, pca.n_components_ + 1)
    cumulative_variance = (
        np.cumsum(pca.explained_variance_ratio_) * 100.0
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        component_numbers,
        cumulative_variance,
        marker="o",
        color="tab:blue",
    )
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA explained variance")
    ax.set_xticks(component_numbers)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    for x_value, y_value in zip(
        component_numbers,
        cumulative_variance,
    ):
        ax.annotate(
            f"{y_value:.1f}%",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    device_index: int,
    n_components: int,
    output_path: Path,
    dpi: int,
) -> None:
    vmin = min(np.min(original), np.min(reconstructed))
    vmax = max(np.max(original), np.max(reconstructed))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(9, 4),
        constrained_layout=True,
    )

    image_original = axes[0].imshow(
        original.T,
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axes[0].set_title(f"Original device {device_index}")
    axes[0].set_xlabel("x pixel")
    axes[0].set_ylabel("y pixel")

    image_reconstructed = axes[1].imshow(
        reconstructed.T,
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axes[1].set_title(
        f"Reconstructed device {device_index}\n"
        f"using {n_components} PCs"
    )
    axes[1].set_xlabel("x pixel")
    axes[1].set_ylabel("y pixel")

    fig.colorbar(
        image_reconstructed,
        ax=axes,
        label="Refractive index",
        shrink=0.85,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    ensemble: np.ndarray,
    pca: PCA,
    coefficients: np.ndarray,
    config: PCAConfig,
) -> None:
    cumulative_variance = np.sum(pca.explained_variance_ratio_)

    print("\nPCA fitting complete.")
    print(f"Input dataset: {config.input_data_path}")
    print(f"Dataset shape: {ensemble.shape}")
    print(f"Number of PCA components: {pca.n_components_}")
    print(f"Coefficient matrix shape: {coefficients.shape}")
    print(
        "Explained variance by selected components: "
        f"{100.0 * cumulative_variance:.3f}%"
    )
    print(f"Saved PCA model: {config.model_output_path}")
    print(f"Saved coefficients: {config.coefficients_output_path}")
    print(f"Saved figures: {config.output_figure_dir}")


def main() -> None:
    config = PCAConfig()
    ensure_output_directories(config)

    ensemble = load_ensemble(config.input_data_path)
    validate_configuration(ensemble, config)

    pca, coefficients, reconstructed_ensemble = fit_pca(
        ensemble,
        config.n_components,
    )

    save_model_outputs(
        pca,
        coefficients,
        reconstructed_ensemble,
        config,
    )

    plot_explained_variance(
        pca,
        config.output_figure_dir / "explained_variance.png",
        config.figure_dpi,
    )

    plot_reconstruction(
        original=ensemble[config.preview_device_index],
        reconstructed=reconstructed_ensemble[config.preview_device_index],
        device_index=config.preview_device_index,
        n_components=config.n_components,
        output_path=config.output_figure_dir / "pca_reconstruction.png",
        dpi=config.figure_dpi,
    )

    print_summary(ensemble, pca, coefficients, config)


if __name__ == "__main__":
    main()
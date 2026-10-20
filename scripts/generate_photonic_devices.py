from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DeviceGenerationConfig:
    lsf_template_path: Path = Path("lumerical/bend_variable_base.lsf")
    output_data_path: Path = Path("data/raw/ensemble_data.npy")
    output_fsp_dir: Path = Path("data/raw/fsp")
    output_preview_path: Path = Path("results/photonics/dataset_extremes.png")
    r_min_um: float = 0.25
    r_max_um: float = 9.0
    num_devices: int = 10
    hide_fdtd: bool = True
    save_fsp: bool = True


def load_lumapi() -> object:
    default_api_path = r"C:\\Program Files\\Lumerical\\v252\\api\\python\\"
    api_path = os.environ.get("LUMERICAL_API_PATH", default_api_path)

    if api_path not in sys.path:
        sys.path.insert(0, api_path)

    try:
        import lumapi
    except ImportError as exc:
        raise ImportError(
            "Could not import lumapi. Check that Lumerical is installed and that "
            f"LUMERICAL_API_PATH points to the folder containing lumapi.py. "
            f"Current path tried: {api_path}"
        ) from exc

    return lumapi


def ensure_parent_dirs(cfg: DeviceGenerationConfig) -> None:
    cfg.output_data_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_fsp_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_preview_path.parent.mkdir(parents=True, exist_ok=True)


def load_lsf_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Lumerical template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_radii_meters(r_min_um: float, r_max_um: float, num_devices: int) -> np.ndarray:
    return np.linspace(r_min_um, r_max_um, num_devices) * 1e-6


def render_lsf_script(template: str, radius_m: float, save_path: Path) -> str:
    script = template.replace("BEND_RADIUS_PLACEHOLDER", f"{radius_m:.12e}")
    script = script.replace("SAVE_PATH_PLACEHOLDER", str(save_path.resolve()).replace("\\", "/"))
    return script


def generate_device_ensemble(cfg: DeviceGenerationConfig) -> tuple[np.ndarray, np.ndarray]:
    lumapi = load_lumapi()
    ensure_parent_dirs(cfg)

    template = load_lsf_template(cfg.lsf_template_path)
    radii_m = build_radii_meters(cfg.r_min_um, cfg.r_max_um, cfg.num_devices)

    device_geometries = []

    print(f"Generating {len(radii_m)} devices from {cfg.r_min_um:.2f} um to {cfg.r_max_um:.2f} um")

    for i, radius_m in enumerate(radii_m, start=1):
        radius_um = radius_m * 1e6
        fsp_path = cfg.output_fsp_dir / f"bend_radius_{radius_um:.2f}um.fsp"
        script = render_lsf_script(template, radius_m, fsp_path)

        with lumapi.FDTD(hide=cfg.hide_fdtd) as fdtd:
            fdtd.eval(script)
            index_map = np.squeeze(fdtd.getdata("index_monitor", "index_x"))
            device_geometries.append(np.real(index_map))

        if not cfg.save_fsp and fsp_path.exists():
            fsp_path.unlink()

        print(f"[{i}/{len(radii_m)}] Generated bend radius = {radius_um:.2f} um")

    ensemble_data = np.stack(device_geometries, axis=0)
    np.save(cfg.output_data_path, ensemble_data)

    print(f"Saved ensemble dataset to: {cfg.output_data_path}")
    print(f"Dataset shape: {ensemble_data.shape}")

    return ensemble_data, radii_m


def save_preview_figure(ensemble_data: np.ndarray, radii_m: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(ensemble_data[0].T, origin="lower", cmap="viridis", aspect="equal")
    axes[0].set_title(f"Minimum radius: {radii_m[0] * 1e6:.2f} um")
    axes[0].set_xlabel("x pixel")
    axes[0].set_ylabel("y pixel")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Index")

    im1 = axes[1].imshow(ensemble_data[-1].T, origin="lower", cmap="viridis", aspect="equal")
    axes[1].set_title(f"Maximum radius: {radii_m[-1] * 1e6:.2f} um")
    axes[1].set_xlabel("x pixel")
    axes[1].set_ylabel("y pixel")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Index")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved dataset preview figure to: {output_path}")


def main() -> None:
    cfg = DeviceGenerationConfig()
    ensemble_data, radii_m = generate_device_ensemble(cfg)
    save_preview_figure(ensemble_data, radii_m, cfg.output_preview_path)


if __name__ == "__main__":
    main()
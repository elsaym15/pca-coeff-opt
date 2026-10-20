from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import scipy.constants as constants
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "models"
LUMERICAL_DIR = ROOT / "lumerical"
RESULTS_DIR = ROOT / "results" / "photonics" / "optimization"
TEMP_DIR = RESULTS_DIR / "temp_sessions"
LOG_DIR = RESULTS_DIR / "logs"

PCA_MODEL_FILE = DATA_DIR / "photonic_pca.joblib"
PCA_COEFF_FILE = DATA_DIR / "photonic_coefficients.npy"
BASE_RUN_LSF = LUMERICAL_DIR / "optimization_base_run.lsf"
BASE_NORUN_LSF = LUMERICAL_DIR / "optimization_base_norun.lsf"

LUMERICAL_API_PATH = os.environ.get(
    "LUMERICAL_API_PATH",
    "C:\\Program Files\\Lumerical\\v252\\api\\python\\",
)

INITIAL_DEVICE_INDEX = 1
OPTIMIZER_METHOD = "SLSQP"
MAXITER = 100
FOM_TOL = 1e-5
USE_PHYSICS_SCALING = False
DYNAMIC_SCALING = True
MODE_NUMBER = "fundamental TM mode"
MODE_DIRECTION = "Backward"
SOURCE_DIRECTION = "Forward"
WAVELENGTH = 1.55e-6
WG_HEIGHT = 0.22e-6


def setup_logging(run_name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_path


def ensure_lumerical_imports() -> Tuple[Any, Any, Any, Any]:
    if LUMERICAL_API_PATH not in sys.path:
        sys.path.insert(0, LUMERICAL_API_PATH)
    try:
        import lumapi
        from lumopt.figures_of_merit.modematch import ModeMatch
        from lumopt.utilities.fields import Fields
        from lumopt.utilities.gradients import GradientFields
        return lumapi, ModeMatch, Fields, GradientFields
    except ImportError as exc:
        raise ImportError(
            "Could not import lumapi/LumOpt. Set LUMERICAL_API_PATH to your Lumerical python API folder."
        ) from exc


@dataclass
class SimWrapper:
    fdtd: Any


@dataclass
class Evaluation:
    raw_fom: float
    physical_transmission: float
    projected_gradient: np.ndarray
    index_map: np.ndarray
    epsilon_map: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunState:
    run_name: str
    run_dir: Path
    temp_dir: Path
    iteration: int = 0
    scaling_factor: Optional[float] = None
    transmission_history: list = field(default_factory=list)
    gradient_norm_history: list = field(default_factory=list)
    coefficients_history: list = field(default_factory=list)
    initial_eval: Optional[Evaluation] = None
    final_eval: Optional[Evaluation] = None


def load_pca_and_coefficients() -> Tuple[Any, np.ndarray]:
    pca = joblib.load(PCA_MODEL_FILE)
    coeffs = np.load(PCA_COEFF_FILE)
    logging.info("Loaded PCA model from %s", PCA_MODEL_FILE)
    logging.info("Loaded coefficient matrix from %s with shape %s", PCA_COEFF_FILE, coeffs.shape)
    return pca, coeffs


def load_lsf(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing LSF file: {path}")
    return path.read_text(encoding="utf-8")


def prepare_run_script(base_run_script: str, temp_dir: Path) -> str:
    temp_dir.mkdir(parents=True, exist_ok=True)
    save_stub = (temp_dir / "monitor_dimensions").as_posix()
    if 'save("Temp_Files\\Monitor_Dimensions");' in base_run_script:
        updated = base_run_script.replace(
            'save("Temp_Files\\Monitor_Dimensions");',
            f'save("{save_stub}");'
        )
        logging.info("Updated run LSF save path for monitor-dimension session to %s", save_stub)
        return updated
    logging.warning("Expected monitor-dimension save line not found in run LSF; using script as-is")
    return base_run_script


def get_monitor_dimensions(run_script: str, lumapi: Any) -> Tuple[np.ndarray, np.ndarray]:
    logging.info("Launching run-LSF dimensions session")
    with lumapi.FDTD(hide=True) as fdtd:
        fdtd.eval(run_script)
        x = np.squeeze(fdtd.getdata("opt_fields", "x"))
        y = np.squeeze(fdtd.getdata("opt_fields", "y"))
    logging.info("Retrieved monitor dimensions: %d x %d", x.size, y.size)
    return np.asarray(x), np.asarray(y)


def inject_geometry(fdtd: Any, index_map: np.ndarray, monitor_x: np.ndarray, monitor_y: np.ndarray) -> None:
    z_geo = np.array([-WG_HEIGHT / 2.0, WG_HEIGHT / 2.0])
    fdtd.putv("n_geo", index_map)
    fdtd.putv("x_geo", monitor_x)
    fdtd.putv("y_geo", monitor_y)
    fdtd.putv("z_geo", z_geo)
    fdtd.eval(
        """
        if (getnamednumber(\"pca_geometry\") > 0) { select(\"pca_geometry\"); delete; }
        if (getnamednumber(\"import\") > 0) { select(\"import\"); delete; }
        addimport;
        set(\"name\", \"pca_geometry\");
        temp = zeros(length(x_geo), length(y_geo), 2);
        temp(:,:,1) = n_geo;
        temp(:,:,2) = n_geo;
        importnk2(temp, x_geo, y_geo, z_geo);
        """
    )


def extract_fields(fdtd: Any, Fields: Any, fom_obj: Any, sim: SimWrapper, wavelength: np.ndarray, apply_scaling: bool) -> Tuple[Any, np.ndarray]:
    x = np.squeeze(fdtd.getresult("opt_fields", "x"))
    y = np.squeeze(fdtd.getresult("opt_fields", "y"))
    z = np.squeeze(fdtd.getresult("opt_fields", "z"))
    nx, ny, nz = x.size, y.size, z.size
    nwl = 1
    e_raw = fdtd.getresult("opt_fields", "E")["E"]
    if apply_scaling:
        scale = fom_obj.get_adjoint_field_scaling(sim)
        e_raw = e_raw * scale
        logging.info("Applied LumOpt field scaling: %s", scale)
    e = e_raw.reshape((nx, ny, nz, nwl, 3))
    index_result = fdtd.getresult("opt_fields_index", "index")
    eps = np.stack(
        (
            np.squeeze(index_result["index_x"] ** 2),
            np.squeeze(index_result["index_y"] ** 2),
            np.squeeze(index_result["index_z"] ** 2),
        ),
        axis=-1,
    ).reshape((nx, ny, nz, nwl, 3))
    d = e * eps * constants.epsilon_0
    fields = Fields(x=x, y=y, z=z, wl=wavelength, E=e, D=d, eps=eps, H=None)
    mag = np.sqrt(np.sum(np.abs(np.squeeze(e)) ** 2, axis=-1)).T
    return fields, mag


def save_session(fdtd: Any, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fdtd.save(str(path))
    logging.info("Saved %s session: %s", label, path)


def evaluate_design(
    coefficients: np.ndarray,
    pca: Any,
    monitor_x: np.ndarray,
    monitor_y: np.ndarray,
    norun_script: str,
    state: RunState,
    lumapi: Any,
    ModeMatch: Any,
    Fields: Any,
    GradientFields: Any,
    tag: str,
) -> Evaluation:
    flattened_index = pca.inverse_transform(np.asarray(coefficients).reshape(1, -1))
    index_map = flattened_index.reshape((monitor_x.size, monitor_y.size))
    epsilon_map = index_map ** 2
    wavelength = np.array([WAVELENGTH])

    logging.info("Iteration %s | Reconstructed index map shape: %s", tag, index_map.shape)

    with lumapi.FDTD(hide=True) as fdtd:
        fdtd.eval(norun_script)
        inject_geometry(fdtd, index_map, monitor_x, monitor_y)
        sim = SimWrapper(fdtd)

        fom_obj = ModeMatch(
            monitor_name="fom",
            mode_number=MODE_NUMBER,
            direction=MODE_DIRECTION,
        )
        fom_obj.initialize(sim)

        if fdtd.getnamednumber("fom_mode_src") > 0:
            fdtd.setnamed("fom_mode_src", "direction", SOURCE_DIRECTION)
            logging.info("Updated fom_mode_src direction to %s", SOURCE_DIRECTION)
        else:
            logging.warning("fom_mode_src not found in layout; continuing without setnamed call")

        fom_obj.make_forward_sim(sim)
        forward_path = state.temp_dir / f"forward_{tag}.fsp"
        save_session(fdtd, forward_path, "forward")
        fdtd.run()

        raw_fom = float(np.squeeze(fom_obj.get_fom(sim)[0]))
        physical_transmission = -raw_fom
        logging.info("Iteration %s | raw_fom=% .8e | transmission=% .8e", tag, raw_fom, physical_transmission)

        forward_fields, forward_mag = extract_fields(fdtd, Fields, fom_obj, sim, wavelength, apply_scaling=False)

        fdtd.switchtolayout()
        if fdtd.getnamednumber("source") > 0:
            fdtd.setnamed("source", "enabled", False)
        fom_obj.make_adjoint_sim(sim)
        adjoint_path = state.temp_dir / f"adjoint_{tag}.fsp"
        save_session(fdtd, adjoint_path, "adjoint")
        fdtd.run()

        adjoint_fields, adjoint_mag = extract_fields(
            fdtd, Fields, fom_obj, sim, wavelength, apply_scaling=USE_PHYSICS_SCALING
        )

        sensitivity_epsilon = GradientFields(forward_fields, adjoint_fields).get_forward_dot_adjoint_center()
        dF_deps = np.asarray(sensitivity_epsilon).flatten()
        d_eps_d_rho = 2.0 * flattened_index.flatten()
        dF_drho = dF_deps * d_eps_d_rho
        projected_gradient = np.asarray(dF_drho @ pca.components_.T).ravel()

        logging.info(
            "Iteration %s | gradient_norm=% .8e | sensitivity_shape=%s | fwd_shape=%s | adj_shape=%s",
            tag,
            np.linalg.norm(projected_gradient),
            np.asarray(sensitivity_epsilon).shape,
            forward_mag.shape,
            adjoint_mag.shape,
        )

        return Evaluation(
            raw_fom=raw_fom,
            physical_transmission=physical_transmission,
            projected_gradient=projected_gradient,
            index_map=index_map,
            epsilon_map=epsilon_map,
            metadata={
                "forward_session": str(forward_path),
                "adjoint_session": str(adjoint_path),
                "forward_field_shape": list(forward_mag.shape),
                "adjoint_field_shape": list(adjoint_mag.shape),
            },
        )


def make_objective(pca: Any, monitor_x: np.ndarray, monitor_y: np.ndarray, norun_script: str, state: RunState,
                   lumapi: Any, ModeMatch: Any, Fields: Any, GradientFields: Any):
    def objective(coefficients: np.ndarray):
        tag = f"iter_{state.iteration:04d}"
        evaluation = evaluate_design(
            coefficients, pca, monitor_x, monitor_y, norun_script, state,
            lumapi, ModeMatch, Fields, GradientFields, tag,
        )
        if state.scaling_factor is None:
            if DYNAMIC_SCALING:
                max_grad = np.max(np.abs(evaluation.projected_gradient))
                state.scaling_factor = 1.0 / max_grad if max_grad > 0 else 1.0
            else:
                state.scaling_factor = 1.0
            logging.info("Initialized optimizer scaling factor: %.8e", state.scaling_factor)

        scaled_value = evaluation.physical_transmission * state.scaling_factor
        scaled_gradient = evaluation.projected_gradient * state.scaling_factor
        state.transmission_history.append(evaluation.physical_transmission)
        state.gradient_norm_history.append(float(np.linalg.norm(evaluation.projected_gradient)))
        state.coefficients_history.append(np.asarray(coefficients).tolist())
        if state.initial_eval is None:
            state.initial_eval = evaluation
        state.iteration += 1
        return scaled_value, scaled_gradient
    return objective


def build_method_options() -> Dict[str, Any]:
    return {"maxiter": MAXITER, "ftol": FOM_TOL, "disp": True}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Photonic PCA optimization with run-LSF dimension check and saved FDTD sessions.")
    parser.add_argument("--device-index", type=int, default=INITIAL_DEVICE_INDEX)
    parser.add_argument("--method", default=OPTIMIZER_METHOD)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"device_{args.device_index}_{args.method.lower()}"
    run_dir = RESULTS_DIR / run_name
    temp_dir = TEMP_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(run_name)

    logging.info("Run directory: %s", run_dir)
    logging.info("Temporary FSP directory: %s", temp_dir)
    logging.info("Using repo paths | data=%s | lumerical=%s", DATA_DIR, LUMERICAL_DIR)

    lumapi, ModeMatch, Fields, GradientFields = ensure_lumerical_imports()
    pca, all_coefficients = load_pca_and_coefficients()
    base_run_script = prepare_run_script(load_lsf(BASE_RUN_LSF), temp_dir)
    norun_script = load_lsf(BASE_NORUN_LSF)
    monitor_x, monitor_y = get_monitor_dimensions(base_run_script, lumapi)

    expected = monitor_x.size * monitor_y.size
    if pca.mean_.size != expected:
        logging.error("PCA feature count %d does not match monitor grid %d x %d = %d", pca.mean_.size, monitor_x.size, monitor_y.size, expected)
        raise ValueError("PCA feature count does not match monitor grid")
    logging.info("Validated PCA feature count against monitor grid")

    initial_coefficients = np.asarray(all_coefficients[args.device_index], dtype=float)
    logging.info("Using initial coefficient row %d with shape %s", args.device_index, initial_coefficients.shape)

    state = RunState(run_name=run_name, run_dir=run_dir, temp_dir=temp_dir)
    objective = make_objective(pca, monitor_x, monitor_y, norun_script, state, lumapi, ModeMatch, Fields, GradientFields)

    result = minimize(
        objective,
        initial_coefficients,
        method=args.method,
        jac=True,
        options=build_method_options(),
    )

    final_coefficients = np.asarray(result.x)
    state.final_eval = evaluate_design(
        final_coefficients, pca, monitor_x, monitor_y, norun_script, state,
        lumapi, ModeMatch, Fields, GradientFields, f"final_{state.iteration:04d}",
    )

    np.save(run_dir / "initial_coefficients.npy", initial_coefficients)
    np.save(run_dir / "final_coefficients.npy", final_coefficients)
    np.save(run_dir / "final_index_map.npy", state.final_eval.index_map)
    np.save(run_dir / "final_epsilon_map.npy", state.final_eval.epsilon_map)

    save_json(run_dir / "run_summary.json", {
        "run_name": run_name,
        "method": args.method,
        "device_index": args.device_index,
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", -1) or -1),
        "fdtd_evaluations": int(state.iteration),
        "initial_transmission": float(state.initial_eval.physical_transmission),
        "final_transmission": float(state.final_eval.physical_transmission),
        "final_gradient_norm": float(np.linalg.norm(state.final_eval.projected_gradient)),
        "scaling_factor": float(state.scaling_factor or 1.0),
        "monitor_shape": [int(monitor_x.size), int(monitor_y.size)],
        "log_file": str(log_path),
        "final_forward_session": state.final_eval.metadata.get("forward_session"),
        "final_adjoint_session": state.final_eval.metadata.get("adjoint_session"),
    })
    save_json(run_dir / "history.json", {
        "transmission": state.transmission_history,
        "gradient_norm": state.gradient_norm_history,
        "coefficients": state.coefficients_history,
    })
    logging.info("Optimization complete | success=%s | message=%s", result.success, result.message)
    logging.info("Saved summary to %s", run_dir / "run_summary.json")


if __name__ == "__main__":
    main()

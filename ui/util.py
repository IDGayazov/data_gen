import json
import os
import sys
import numpy as np
import pandas as pd


def normalize(X):
    """Per-column min-max normalization used for classification inference."""
    if isinstance(X, pd.DataFrame):
        X = X.values

    X_norm = X.copy()

    for i in range(X.shape[1]):
        col = X[:, i]
        col_min = np.min(col)
        col_max = np.max(col)
        col_range = col_max - col_min

        if col_range != 0:
            X_norm[:, i] = (col - col_min) / col_range
        else:
            X_norm[:, i] = 0

    return X_norm


def predict(ort_session, df, k, config_file):
    """Classification inference — returns predicted class name string."""
    t_D = df["t_D"]
    y = df["P_wD_gauss"]
    dy = df["dP_wD"]

    t_D = t_D / k
    dy = dy / k

    data = np.column_stack([dy, t_D])
    data_norm = normalize(data)

    input_data = np.expand_dims(data_norm, axis=0).astype("float32")

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    predictions = ort_session.run([output_name], {input_name: input_data})[0]
    predicted_class_idx = np.argmax(predictions, axis=1)[0]

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    class_names = config.get('class_names', [])
    return class_names[predicted_class_idx]


def predict_params(ort_session, df, k, scaler_config_path):
    """
    Regression inference — returns {param_name: value} dict,
    or None if scaler config file does not exist yet.
    """
    if not os.path.exists(scaler_config_path):
        return None

    with open(scaler_config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    t_D = df['t_D'].values.astype(np.float64)
    dy = df['dP_wD'].values.astype(np.float64)

    # 1. Divide by k (same as training preprocessing)
    t_D_k = t_D / k
    dy_k = dy / k

    # 2. Stack [dP_wD, t_D] — same column order as training
    data = np.column_stack([dy_k, t_D_k])  # (N, 2)

    # 3. log10 with eps guard
    eps = 1e-10
    data = np.log10(np.abs(data) + eps)  # (N, 2)

    # 4. Global standardize using saved X_mean / X_std (shape (1,1,2))
    X_mean = np.array(cfg['X_mean'])  # stored as (1,1,2) nested list
    X_std = np.array(cfg['X_std'])
    data = (data - X_mean.squeeze(0)) / X_std.squeeze(0)  # (N, 2)

    # 5. Add batch dim → (1, N, 2)
    input_data = np.expand_dims(data, axis=0).astype(np.float32)

    # 6. ONNX inference
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    predictions = ort_session.run([output_name], {input_name: input_data})[0]
    y_pred = predictions[0]  # (n_params,)

    # 7. Inverse KValueScaler transform
    output_names = cfg['output_names']
    n_cols = len(output_names)
    log_cols = cfg['log_cols']
    other_cols = cfg['other_cols']

    y_inv = np.zeros(n_cols, dtype=np.float64)

    # Column 0 (k): inverse StandardScaler then /k_multiplier
    k_mean = float(np.array(cfg['k_scaler_mean'])[0])
    k_scale = float(np.array(cfg['k_scaler_scale'])[0])
    y_inv[0] = (float(y_pred[0]) * k_scale + k_mean) / cfg['k_multiplier']

    # log_cols: inverse StandardScaler then 10^x
    for col in log_cols:
        log_mean = float(np.array(cfg['log_scaler_mean'][str(col)])[0])
        log_scale = float(np.array(cfg['log_scaler_scale'][str(col)])[0])
        y_inv[col] = 10.0 ** (float(y_pred[col]) * log_scale + log_mean)

    # other_cols: inverse StandardScaler
    other_mean = np.array(cfg['other_scaler_mean'])
    other_scale = np.array(cfg['other_scaler_scale'])
    for i, col in enumerate(other_cols):
        y_inv[col] = float(y_pred[col]) * float(other_scale[i]) + float(other_mean[i])

    return {name: float(y_inv[i]) for i, name in enumerate(output_names)}


# class_name → file name suffix (shared across all model families)
_REG_MODEL_SUFFIX = {
    'homogeneous_inf':       'homogeneous_inf',
    'homogeneous_fin':       'homogeneous_fin',
    'dual_porosity_inf':     'dual_porosity_inf',
    'dual_porosity_fin':     'dual_porosity_fin',
    'dual_permeability_inf': 'dual_permeability_inf',
    'dual_permeability_fin': 'dual_permeability_fin',
    'radial_composite_inf':  'radial_composite',
}

# model_key → file name prefix
_REG_MODEL_PREFIX = {
    'cnn':      '1d_cnn_model_regression_',
    'lstm':     'lstm_model_regression_',
    'cnn_lstm': 'cnn_lstm_model_regression_',
}


def get_regression_model_paths(class_name, models_dir, model_key='cnn'):
    """Returns (onnx_path, scaler_config_path) for a predicted class name and model family."""
    suffix = _REG_MODEL_SUFFIX.get(class_name)
    prefix = _REG_MODEL_PREFIX.get(model_key, _REG_MODEL_PREFIX['cnn'])
    if suffix is None:
        return None, None
    base = prefix + suffix
    onnx_path = os.path.join(models_dir, base + '.onnx')
    scaler_path = os.path.join(models_dir, base + '_scaler_config.json')
    return onnx_path, scaler_path


def build_predicted_curve(class_name, params_dict, t_D_array):
    """
    Builds a theoretical pressure curve from predicted parameters.
    Returns DataFrame with columns [t_D, P_wD, dP_wD], or None on error.
    """
    # Make sure project root is on sys.path for model imports
    _root = os.path.join(os.path.dirname(__file__), '..')
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from inversion.shtefest_algorithm import ShtefestAlgorithm
    alg = ShtefestAlgorithm(N=12)

    try:
        if class_name == 'homogeneous_inf':
            from model.homogeneous.infinite_homogeneous_model import InfiniteHomogeneousReservoirModel
            model = InfiniteHomogeneousReservoirModel(
                C_D=params_dict['C_D'],
                S=params_dict['S'],
            )

        elif class_name == 'homogeneous_fin':
            from model.homogeneous.finite_homogeneous_model import FiniteHomogeneousReservoirModel
            model = FiniteHomogeneousReservoirModel(
                C_D=params_dict['C_D'],
                S=params_dict['S'],
                R_D_E=params_dict['r_D_e'],
            )

        elif class_name == 'dual_porosity_inf':
            from model.dualporosity.infinite_dual_porosity_model import InfiniteDualPorosityReservoirModel
            model = InfiniteDualPorosityReservoirModel(
                C_D=params_dict['C_D'],
                S=params_dict['S'],
                omega=params_dict['omega'],
                lam=params_dict['lambda'],
            )

        elif class_name == 'dual_porosity_fin':
            from model.dualporosity.finite_dual_porosity_model import FiniteDualPorosityReservoirModel
            model = FiniteDualPorosityReservoirModel(
                C_D=params_dict['C_D'],
                S=params_dict['S'],
                omega=params_dict['omega'],
                lam=params_dict['lambda'],
                R_D_e=params_dict['r_D_e'],
            )

        elif class_name == 'dual_permeability_inf':
            from model.dualpermeability.infinite_dual_permeability_model import InfiniteDualPermeabilityReservoirModel
            model = InfiniteDualPermeabilityReservoirModel(
                C_D=params_dict['C_D'],
                S=params_dict['S1'],
                omega=params_dict['omega'],
                lam=params_dict['lambda'],
                kappa=params_dict['kappa'],
            )

        elif class_name == 'dual_permeability_fin':
            from model.dualpermeability.finite_dual_permeability_model import FiniteDualPermeabilityReservoirModel
            model = FiniteDualPermeabilityReservoirModel(
                C_D=params_dict['C_D'],
                S=params_dict['S1'],
                omega=params_dict['omega'],
                lam=params_dict['lambda'],
                kappa=params_dict['kappa'],
                R_D_E=params_dict['r_D_e'],
            )

        elif class_name == 'radial_composite_inf':
            from model.radialcomposite.infinite_radial_composite_model import InfiniteRadialCompositeReservoirModelV2
            S = params_dict.get('S', params_dict.get('S1', 0.0))
            model = InfiniteRadialCompositeReservoirModelV2(
                C_D=params_dict['C_D'],
                S=S,
                M12=params_dict.get('M12', 1.0),
                omega12=params_dict.get('omega12', 1.0),
                r_fD=params_dict['r_fD'],
            )

        else:
            return None

        result = model.pressure(t_D_array, alg)
        result.derivative(smoothig_alg='regression', delta=0.175)
        return result.get_pressure()

    except Exception as e:
        print(f"build_predicted_curve error ({class_name}): {e}")
        return None


def format_params(params_dict):
    """Returns a human-readable string of predicted parameters."""
    lines = []
    for name, val in params_dict.items():
        if abs(val) < 1e-3 or abs(val) > 1e4:
            lines.append(f"  {name}: {val:.3e}")
        else:
            lines.append(f"  {name}: {val:.4f}")
    return "\n".join(lines)

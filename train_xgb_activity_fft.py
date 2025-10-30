"""Train an XGBoost multiclass classifier for activity detection.

This script:
- reads `QOM/gait_labels_qom.csv` for labels and file paths
- loads each trial CSV from `User_Data_Labelled/`
- extracts simple time-series features from `Voltage(V)` per file
- trains an XGBoost classifier and evaluates it
- saves the trained model and a small report to `models/`

Usage (from project root):
    python train_xgb_activity.py

Optional args: --data-dir, --labels, --output-dir, --test-size, --random-state
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

try:
    import xgboost as xgb
except Exception as e:
    raise ImportError("xgboost is required. Install with `pip install xgboost`.") from e


def extract_features_from_voltage(
    voltage: np.ndarray,
    time: np.ndarray | None = None
) -> dict:
    """Compute a small set of features from a 1D voltage array."""
    if voltage.size == 0:
        return {}
    feats = {}
    feats['mean'] = float(np.mean(voltage))
    feats['std'] = float(np.std(voltage))
    feats['min'] = float(np.min(voltage))
    feats['max'] = float(np.max(voltage))
    # feats['median'] = float(np.median(voltage))
    feats['skew'] = float(stats.skew(voltage))
    feats['kurtosis'] = float(stats.kurtosis(voltage))
    feats['energy'] = float(np.sum(np.square(voltage)))
    # simple spectral proxy: mean abs FFT magnitude (coarse)
    try:
        fft_mag = np.abs(np.fft.fft(voltage))
        feats['fft_mean'] = float(np.mean(fft_mag))
        feats['fft_max'] = float(np.max(fft_mag))
    except Exception:
        feats['fft_mean'] = 0.0
        feats['fft_max'] = 0.0

    if time is not None:
        # Compute additional features based on time
        peak_idxs, _ = find_peaks(voltage, height=0)
        feats['stride_count'] = len(peak_idxs)
        feats['mean_stride_duration'] = float(np.mean(np.diff(time[peak_idxs]))) if len(peak_idxs) > 1 else 0.0

    return feats

def extract_frequency_features(
    signal: np.ndarray,
    labels: list[str],
    sampling_rate: int | float,
    window_size: int,
    step_size: int | float,
):
    """
    Extracts frequency-domain features from a time-series signal using a sliding window.
    """
    features = []
    window_labels = []

    # Slide a window across the signal
    for start in range(0, len(signal) - window_size + 1, step_size):
        end = start + window_size
        window_data = signal[start:end]

        # --- FFT Calculation ---
        N = len(window_data)
        fft_complex = np.fft.fft(window_data)
        fft_magnitude = np.abs(fft_complex[:N // 2])
        frequencies = np.fft.fftfreq(N, 1 / sampling_rate)[:N // 2]

        # --- Feature Extraction for the window ---
       
        # 1. Dominant Frequency (ignoring the 0 Hz DC component)
        if len(fft_magnitude[1:]) > 0:
            peak_index = np.argmax(fft_magnitude[1:]) + 1
            dominant_frequency = frequencies[peak_index]
        else:
            dominant_frequency = 0

        # 2. Spectral Energy (sum of squared magnitudes)
        spectral_energy = np.sum(fft_magnitude**2) / N

        # 3. Spectral Centroid
        # Weighted average of frequencies, indicates "center of mass" of the spectrum
        # Avoid division by zero if the spectrum is all zeros
        if np.sum(fft_magnitude) > 0:
            spectral_centroid = np.sum(frequencies * fft_magnitude) / np.sum(fft_magnitude)
        else:
            spectral_centroid = 0

        # Store the features for this window
        features.append([dominant_frequency, spectral_energy, spectral_centroid])
       
        # Assign a label to the window (e.g., the label at the end of the window)
        # window_labels.append(labels[end - 1])

    # Create a DataFrame
    feature_df = pd.DataFrame(
        features,
        columns=['dominant_freq', 'spectral_energy', 'spectral_centroid']
    )
    feature_df['activity'] = labels
   
    return feature_df

def load_dataset(labels_path: Path, data_dir: Path):
    labels = pd.read_csv(labels_path)

    # Build a mapping of available files in the data dir (lowercased) -> real path
    available = {p.name.lower(): p for p in data_dir.glob('*.csv')}

    rows = []
    missing = []
    for _, r in labels.iterrows():
        fp = str(r['file_path'])
        fp_norm = fp.lower()
        if fp_norm in available:
            fullpath = available[fp_norm]
            try:
                df = pd.read_csv(fullpath)
            except Exception as e:
                missing.append((fp, f"read_error:{e}"))
                continue

            # Try to find the voltage column; be flexible to small name changes
            col_candidates = [c for c in df.columns if 'volt' in c.lower()]
            time_col_candidates = [c for c in df.columns if 'time' in c.lower()]
            if len(col_candidates) == 0:
                missing.append((fp, 'no_voltage_column'))
                continue
            volt_col = col_candidates[0]
            time_col = time_col_candidates[0]
            voltage = df[volt_col].dropna().values.astype(float)
            time = df[time_col].dropna().values.astype(float)

            feats = extract_features_from_voltage(voltage, time)
            feats['file_path'] = fp
            feats['activity'] = r['activity']
            # feats['height'] = r['height_cm']
            # feats['weight'] = r['weight_kg']
            if feats['activity'] == 'stand':
                continue
            feats['subject_id'] = r.get('subject_id', '')
            rows.append(feats)
        else:
            missing.append((fp, 'not_found'))

    feat_df = pd.DataFrame(rows)
    scaler = StandardScaler()
    # ht_scaled = scaler.fit_transform(feat_df[['height', 'weight']])
    # feat_df['height'] = ht_scaled[:, 0]
    # feat_df['weight'] = ht_scaled[:, 1]
    breakpoint()
    return feat_df, missing


def load_dataset_all(labels_path: Path, data_dir: Path):
    labels = pd.read_csv(labels_path)

    # Build a mapping of available files in the data dir (lowercased) -> real path
    available = {p.name.lower(): p for p in data_dir.glob('*.csv')}

    rows = []
    missing = []
    all_dfs = []
    for _, r in labels.iterrows():
        if r['activity'] == 'stand':
            continue
        fp = str(r['file_path'])
        fp_norm = fp.lower()
        if fp_norm in available:
            fullpath = available[fp_norm]
            try:
                df = pd.read_csv(fullpath)
            except Exception as e:
                missing.append((fp, f"read_error:{e}"))
                continue

            # Try to find the voltage column; be flexible to small name changes
            volt_col_candidates = [c for c in df.columns if 'volt' in c.lower()]
            time_col_candidates = [c for c in df.columns if 'time' in c.lower()]
            if len(volt_col_candidates) == 0:
                missing.append((fp, 'no_voltage_column'))
                continue
            if len(time_col_candidates) == 0:
                missing.append((fp, 'no_time_column'))
                continue
            volt_col = volt_col_candidates[0]
            time_col = time_col_candidates[0]
            voltage = df[volt_col].dropna().values.astype(float)
            time = df[time_col].dropna().values.astype(float)
            freq_feats_df = extract_frequency_features(
                voltage,
                r['activity'],
                sampling_rate=128,
                window_size=1024,
                step_size=10
            )
            activity = r['activity']

            _df = pd.DataFrame({'time': time, 'voltage': voltage, 'activity': activity})
            all_dfs.append(freq_feats_df)


            feats = extract_features_from_voltage(voltage)
            feats['file_path'] = fp
            feats['activity'] = r['activity']
            feats['subject_id'] = r.get('subject_id', '')
            rows.append(feats)
        else:
            missing.append((fp, 'not_found'))

    feat_df_all = pd.concat(all_dfs, ignore_index=True)
    feat_df = pd.DataFrame(rows)
    # breakpoint()
    return feat_df_all, missing


def train_and_evaluate(X, y, output_dir: Path, args):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=args.test_size, random_state=args.random_state, stratify=y_enc
    )

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=1000,
        max_depth=6,
        reg_lambda=1,
        eta=0.3,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'label_encoder': le}, output_dir / 'xgb_activity_model.joblib')

    with open(output_dir / 'report.txt', 'w') as f:
        f.write(f'Accuracy: {acc}\n\n')
        f.write(report)

    # Save a small confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.savetxt(output_dir / 'confusion_matrix.csv', cm, delimiter=',', fmt='%d')

    # Feature importance (scikit-learn style)
    try:
        importances = model.feature_importances_
        feat_names = X.columns.tolist()
        fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        fi.to_csv(output_dir / 'feature_importances.csv')
    except Exception:
        pass

    return acc, report


def parse_args():
    p = argparse.ArgumentParser(description='Train XGBoost multiclass for activity detection')
    p.add_argument('--data-dir', type=Path, default=Path('User_Data_Labelled'), help='Directory with trial CSVs')
    p.add_argument('--labels', type=Path, default=Path('QOM/gait_labels_qom.csv'), help='Master labels CSV')
    p.add_argument('--output-dir', type=Path, default=Path('models'), help='Where to write model and reports')
    p.add_argument('--test-size', type=float, default=0.25)
    p.add_argument('--random-state', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print('Loading dataset...')
    feat_df, missing = load_dataset_all(args.labels, args.data_dir)

    print(f'Loaded {len(feat_df)} feature rows; {len(missing)} missing / unreadable entries')
    if len(missing) > 0:
        print('Examples of missing entries:', missing[:5])
        breakpoint()

    if feat_df.empty:
        print('No features to train on. Exiting.')
        return

    # Prepare X, y
    X = feat_df.drop(columns=['file_path', 'activity', 'subject_id'], errors='ignore')
    # X = X.drop(columns=["fft_mean", "fft_max"])
    y = feat_df['activity']

    print('Training XGBoost classifier...')
    acc, report = train_and_evaluate(X, y, args.output_dir, args)

    print(f'Training complete. Test accuracy: {acc:.4f}')
    print('Detailed classification report:')
    print(report)
    print(f'Model and artifacts saved to {args.output_dir}')


if __name__ == '__main__':
    main()

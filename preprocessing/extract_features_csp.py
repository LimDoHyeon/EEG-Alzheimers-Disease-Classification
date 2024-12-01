import os
import mne
from mne.decoding import CSP
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.signal import welch
from antropy import spectral_entropy
from sklearn.preprocessing import StandardScaler

mne.set_log_level('warning')


def compute_band_power(data, sfreq, bands):
    """
    EEG 데이터의 주파수 대역별 Band Power를 계산합니다.

    Args:
        data (np.ndarray): (n_channels, n_samples) 형태의 EEG 데이터.
        sfreq (float): 샘플링 주파수 (Hz).
        bands (dict): 주파수 대역 이름과 범위를 매핑한 딕셔너리.

    Returns:
        dict: 주파수 대역별 Band Power.
    """
    band_powers = {}
    freqs, psd = welch(data, fs=sfreq, nperseg=sfreq*2, axis=1)
    for band_name, (fmin, fmax) in bands.items():
        band_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.sum(psd[:, band_idx], axis=1)  # 각 채널별로 계산
        band_powers[band_name] = band_power
    return band_powers


def edf_to_epochs(edf_path, channels=None, chunk_size=None, labels=None):  # CSP
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    chunk_size=30*250

    # 채널 선택
    if channels:
        raw.pick_channels(channels)

    data = raw.get_data()  # (n_channels, n_samples)
    sfreq = raw.info['sfreq']

    # Trial 단위로 분할
    n_channels, n_samples = data.shape
    if chunk_size is None:
        chunk_size = int(sfreq * 30)  # 기본값: 30초
    n_trials = n_samples // chunk_size
    data = data[:, :n_trials * chunk_size]  # 나머지 자름
    data = data.reshape((n_trials, n_channels, chunk_size))  # 3D 변환

    # 레이블 설정
    if labels is None or len(labels) != n_trials:
        raise ValueError(f"Number of labels ({len(labels)}) must match number of trials ({n_trials}).")

    # Info 객체 생성 및 Epochs 생성
    info = mne.create_info(ch_names=raw.ch_names, sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(data, info)

    return epochs, labels


def transform_labels_to_trials(labels, chunk_size):  # CSP
    n_samples = len(labels)
    n_trials = n_samples // chunk_size

    # Trial 단위로 나머지 제거
    labels = labels[:n_trials * chunk_size].reshape(n_trials, chunk_size)

    # 각 Trial의 레이블 결정 (예: 평균값 기준)
    trial_labels = np.round(labels.mean(axis=1)).astype(int)  # 예: 평균값 기준
    return trial_labels


def extract_features_binary(file_path, label, channels, normalize_kurtosis=True):
    """
    단일 EDF 파일에서 특징 추출 후 데이터프레임으로 반환
    """
    # EDF 파일 로드
    raw = mne.io.read_raw_edf(file_path, preload=True)
    if channels:
        raw.pick_channels(channels)

    data = raw.get_data()  # (n_channels, n_samples)
    sfreq = raw.info['sfreq']

    features = {}
    # 시간 영역 특징
    kurtosis_values = [kurtosis(data[ch_idx]) for ch_idx in range(data.shape[0])]
    if normalize_kurtosis:
        scaler = StandardScaler()  # Min-Max 스케일링 (0~1)
        kurtosis_values = scaler.fit_transform(np.array(kurtosis_values).reshape(-1, 1)).flatten()
    for ch_idx, ch_name in enumerate(raw.ch_names):
        features[f"kurtosis_{ch_name}"] = kurtosis_values[ch_idx]

    """
    # 주파수 대역 특징
    band_powers = compute_band_power(data, sfreq, bands)
    for band_name, power_values in band_powers.items():
        for ch_idx, ch_name in enumerate(raw.ch_names):
            features[f"{band_name}_{ch_name}"] = power_values[ch_idx]
    """

    # Spectral Entropy
    for ch_idx, ch_name in enumerate(raw.ch_names):
        features[f"spectral_entropy_{ch_name}"] = spectral_entropy(data[ch_idx], sfreq, method='welch')

    # DataFrame 생성
    feature_df = pd.DataFrame([features])

    # label 추가 및 마지막 열로 이동
    feature_df['label'] = label
    feature_df = feature_df[[col for col in feature_df.columns if col != 'label'] + ['label']]

    return feature_df


def extract_csp(data_dir, label, channels):
    labels = np.load(label)
    chunk_size= 30*250

    y = transform_labels_to_trials(labels, chunk_size=30*250)
    print(f"Labels shape: {y.shape}")

    # MNE Epochs 객체로 변환
    epochs, _ = edf_to_epochs(data_dir, channels=channels, chunk_size=chunk_size, labels=y)
    X = epochs.get_data()  # (n_trials, n_channels, n_times)
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    X_new = csp.fit_transform(X, y)
    return X_new


def process_group(group_path, label, channels, bands):
    """
    그룹(AD, MCI, NC)의 모든 EDF 파일에 대해 특징 추출
    """
    feature_frames = []
    for subfolder in os.listdir(group_path):
        subfolder_path = os.path.join(group_path, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith(".edf"):
                    edf_path = os.path.join(subfolder_path, file)
                    print(f"Processing {edf_path}")
                    feature_df = extract_features_binary(edf_path, label, channels, bands)
                    feature_frames.append(feature_df)
    return pd.concat(feature_frames, ignore_index=True)


def main():
    base_dir = '/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/'
    edf_dir = '/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/extracted/ver4 - edf4CSP/merged_nc2.edf'
    csp_label_dir = '/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/extracted/ver4 - edf4CSP/merged_nc2_labels.npy'
    output_csv = '/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/extracted/ver4 - edf4CSP/features_nc2.csv'  # 저장할 CSV 경로
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3',
                'Pz', 'P4', 'P8', 'O1', 'O2']
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }

    # 아래 주석은 필요에 따라 활성화 (AD vs NC 분류 데이터셋이 필요하면 첫 번째 주석 해제)
    # groups = {"AD": 1, "NC": 0}  # 그룹과 레이블 매핑
    # groups = {"MCI": 1, "NC": 0}  # 그룹과 레이블 매핑
    # groups = {"NC2": 0}  # 그룹과 레이블 매핑
    all_features = []

    for group_name, label in groups.items():
        group_path = os.path.join(base_dir, group_name)
        group_features = process_group(group_path, label, channels, bands)
        all_features.append(group_features)

    # 모든 그룹 데이터를 하나로 병합
    final_df = pd.concat(all_features, ignore_index=True)

    # final_df에서 label열을 따로 분리
    labels = final_df.pop('label')

    # Common Spatial Pattern
    csp = extract_csp(edf_dir, csp_label_dir, channels)
    # csp값을 feature에 추가
    for i in range(csp.shape[1]):
        final_df[f'csp{i}'] = csp[:, i]

    # 분리한 label열을 다시 추가
    final_df['label'] = labels
    final_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
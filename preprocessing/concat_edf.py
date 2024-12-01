import os
import mne
import numpy as np
from concurrent.futures import ProcessPoolExecutor

mne.set_log_level('warning')


def load_edf_and_preprocess(file_path, channels=None, chunk_size=None):
    """
    단일 EDF 파일을 로드하고 청크 단위로 데이터를 전처리합니다.

    Args:
        file_path (str): .edf 파일 경로.
        channels (list, optional): 관심 채널 리스트. 선택적으로 지정.
        chunk_size (int, optional): 청크의 샘플 크기.

    Returns:
        list: 청크로 나뉜 데이터 배열 리스트. 각 청크는 (n_channels, n_samples) 형태.
    """
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # 채널 선택
    if channels:
        raw.pick_channels(channels)

    # 데이터 가져오기
    data = raw.get_data()  # (n_channels, n_samples)

    # 데이터를 청크로 나누기
    n_samples = data.shape[1]
    if chunk_size:
        chunks = [
            data[:, i:i + chunk_size]
            for i in range(0, n_samples, chunk_size)
        ]
    else:
        chunks = [data]

    return chunks


def load_and_merge_group(group_path, label, channels=None, chunk_size=None):
    """
    그룹 내 모든 .edf 파일을 청크 단위로 처리하여 병합하고 라벨을 추가합니다.

    Args:
        group_path (str): 그룹 폴더 경로.
        label (int): 그룹 라벨 (예: 1: MCI, 0: NC).
        channels (list, optional): 관심 채널 리스트.
        chunk_size (int, optional): 청크의 샘플 크기.

    Returns:
        tuple: 병합된 데이터 배열 (n_channels, n_samples)과 해당 라벨 리스트.
    """
    all_chunks = []
    for subfolder in os.listdir(group_path):
        subfolder_path = os.path.join(group_path, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith('.edf'):
                    edf_path = os.path.join(subfolder_path, file)
                    chunks = load_edf_and_preprocess(edf_path, channels, chunk_size)
                    all_chunks.extend(chunks)

    # 데이터 병합
    if len(all_chunks) > 0:
        merged_data = np.concatenate([chunk.astype(np.float32) for chunk in all_chunks], axis=1)
        labels = np.full(merged_data.shape[1], label)  # 라벨 생성
    else:
        merged_data = np.array([])  # 빈 경우
        labels = np.array([])

    return merged_data, labels


def process_dataset_parallel(base_dir, channels=None, chunk_size=None, max_workers=4):
    """
    병렬로 데이터를 처리하여 그룹별 병합하고 라벨링합니다.

    Args:
        base_dir (str): AD, MCI, NC 그룹 폴더가 포함된 상위 디렉토리 경로.
        channels (list, optional): 관심 채널 리스트.
        chunk_size (int, optional): 청크의 샘플 크기.
        max_workers (int, optional): 병렬 처리 시 사용될 최대 프로세스 수.

    Returns:
        tuple: 병합된 데이터 배열과 라벨 배열.
    """
    # groups = {'AD': 2, 'NC': 0}  # 그룹별 라벨
    groups = {'NC2': 0}  # 그룹별 라벨
    merged_data_list = []
    label_list = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(load_and_merge_group, os.path.join(base_dir, group), label, channels, chunk_size): group
            for group, label in groups.items()
        }
        for future in futures:
            group = futures[future]
            try:
                data, labels = future.result()
                merged_data_list.append(data)
                label_list.append(labels)
                print(f"{group} 데이터 처리 완료: {data.shape}")
            except Exception as e:
                print(f"{group} 데이터 처리 중 오류 발생: {e}")

    # 병합된 데이터와 라벨
    if merged_data_list:
        final_data = np.hstack(merged_data_list)  # 데이터 병합
        final_labels = np.hstack(label_list)  # 라벨 병합
    else:
        final_data = np.array([])
        final_labels = np.array([])

    return final_data, final_labels


def save_merged_as_edf_concat_with_labels(data, labels, output_file, sfreq, channels):
    """
    병합된 데이터를 라벨과 함께 EDF 파일로 저장합니다.

    Args:
        data (np.ndarray): 병합된 데이터 배열 (n_channels, n_samples).
        labels (np.ndarray): 병합된 라벨 배열.
        output_file (str): 저장할 EDF 파일 경로.
        sfreq (float): 샘플링 주파수.
        channels (list): 기본 EEG 채널 이름 리스트.
    """
    # 채널 정보 생성
    ch_types = ['eeg'] * data.shape[0]
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=ch_types)

    print('Raw 객체 생성')
    raw = mne.io.RawArray(data, info)

    print('EDF 파일로 저장')
    raw.export(output_file, fmt='edf', overwrite=True)
    print(f"EDF 파일 저장 완료: {output_file}")

    # 라벨 저장
    label_file = output_file.replace('.edf', '_labels.npy')
    np.save(label_file, labels)
    print(f"라벨 저장 완료: {label_file}")


def main():
    base_dir = '/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/'
    output_dir = '/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/extracted/ver4 - edf4CSP'
    output_file = os.path.join(output_dir, "merged_nc2.edf")  # 정확한 파일 경로

    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3',
                'Pz', 'P4', 'P8', 'O1', 'O2']
    chunk_size = 250 * 30  # 30초 기준으로 데이터 청크 분할
    sfreq = 250  # 샘플링 주파수

    # 병렬로 데이터 병합 및 라벨링
    data, labels = process_dataset_parallel(base_dir, channels, chunk_size)
    print(f"병합된 데이터 크기: {data.shape}, 라벨 크기: {labels.shape}")

    # 병합된 데이터를 라벨과 함께 EDF 파일로 저장
    save_merged_as_edf_concat_with_labels(data, labels, output_file, sfreq, channels)


if __name__ == "__main__":
    main()

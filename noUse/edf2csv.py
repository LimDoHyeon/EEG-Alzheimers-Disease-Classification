import os
import mne
import pandas as pd


def edf2csv(edf_path, csv_path):
    """EDF 파일을 CSV 파일로 변환."""
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    # 채널 데이터와 시간 정보, 채널 이름 가져오기
    data, times = raw.get_data(return_times=True)
    channel_names = raw.ch_names

    # 데이터프레임 생성 (시간 + 채널 데이터)
    df = pd.DataFrame(data.T, columns=channel_names)  # 데이터를 Transpose하여 채널을 열로 만듦
    df.insert(0, 'Time (s)', times)  # 첫 번째 열에 시간 추가

    # CSV 파일로 저장
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)  # 저장 경로에 폴더 생성
    df.to_csv(csv_path, index=False)
    print(f"CSV 파일 저장 완료: {csv_path}")


def convert_all_edf_to_csv(ad_dir, ad_csv_dir):
    """AD 디렉토리 내부의 모든 EDF 파일을 AD_csv 디렉토리에 변환하여 저장."""
    for root, dirs, files in os.walk(ad_dir):
        for file in files:
            if file.endswith(".edf"):
                edf_path = os.path.join(root, file)  # 원본 EDF 파일 경로
                relative_path = os.path.relpath(edf_path, ad_dir)  # AD 디렉토리 기준 상대 경로
                csv_path = os.path.join(ad_csv_dir, os.path.splitext(relative_path)[0] + ".csv")  # CSV 파일 경로

                # EDF -> CSV 변환
                edf2csv(edf_path, csv_path)


if __name__ == "__main__":
    mci_dir = "/preprocessSeg/data/MCI"
    mci_csv_dir = "/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/MCI_csv"
    convert_all_edf_to_csv(mci_dir, mci_csv_dir)

    # nc_dir = "/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/NC"
    # nc_csv_dir = "/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/NC_csv"
    # convert_all_edf_to_csv(nc_dir, nc_csv_dir)

import os
import pandas as pd


def merge_csvs_in_folder(folder_path, output_folder):
    """
    지정된 폴더 내 모든 CSV 파일을 병합하여 새로운 CSV로 저장.

    Args:
        folder_path (str): 병합 대상 CSV 파일들이 있는 폴더 경로.
        output_folder (str): 병합된 CSV 파일이 저장될 폴더 경로.
    """
    folder_name = os.path.basename(folder_path.rstrip('/'))  # 폴더 이름 추출
    output_file = os.path.join(output_folder, f"{folder_name}.csv")  # 출력 파일 경로

    # 병합 대상 파일 리스트 가져오기
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        print(f"폴더 '{folder_name}'에 CSV 파일이 없습니다. 스킵합니다.")
        return

    # 모든 CSV 파일 병합
    df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)

    # 출력 폴더 생성 (필요하면)
    os.makedirs(output_folder, exist_ok=True)

    # 병합된 CSV 저장
    merged_df.to_csv(output_file, index=False)
    print(f"폴더 '{folder_name}'의 CSV 파일이 병합되어 저장되었습니다: {output_file}")


def merge_all_subfolders(ad_dir, output_dir):
    """
    AD 폴더 하위의 모든 폴더에서 CSV 파일을 병합하여 저장.

    Args:
        ad_dir (str): AD 디렉토리 경로.
    """
    # AD 디렉토리 상위 경로
    base_dir = os.path.dirname(ad_dir.rstrip('/'))

    # 병합된 파일 저장 경로
    output_dir = os.path.join(base_dir, output_dir)

    # AD 디렉토리 하위 폴더 반복 처리
    for subfolder in os.listdir(ad_dir):
        subfolder_path = os.path.join(ad_dir, subfolder)

        if os.path.isdir(subfolder_path):  # 디렉토리인지 확인
            merge_csvs_in_folder(subfolder_path, output_dir)


if __name__ == "__main__":
    # ad_dir = "/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/Concatenated/AD_concat"
    # ad_output_dir = "Concatenated"
    # merge_all_subfolders(ad_dir, ad_output_dir)

    mci_dir = "/preprocessSeg/Concatenated/MCI_concat"
    mci_output_dir = "/preprocessSeg/Concatenated"
    merge_all_subfolders(mci_dir, mci_output_dir)

    # nc_dir = "/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/Concatenated/NC_concat"
    # nc_output_dir = "Concatenated"
    # merge_all_subfolders(nc_dir, nc_output_dir)

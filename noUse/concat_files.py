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


if __name__ == "__main__":
    # ad_dir = "/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/Concatenated/AD_concat"
    # ad_output_dir = "/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/Concatenated"
    # merge_csvs_in_folder(ad_dir, ad_output_dir)

    # mci_dir = "/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/Concatenated/MCI_concat"
    # mci_output_dir = "/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/Concatenated"
    # merge_csvs_in_folder(mci_dir, mci_output_dir)

    nc_dir = "/preprocessSeg/Concatenated/NC_concat"
    nc_output_dir = "Concatenated"
    merge_csvs_in_folder(nc_dir, nc_output_dir)

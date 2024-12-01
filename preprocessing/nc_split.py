import os
import shutil


# 폴더 이름 생성 함수
def generate_folder_name(index):
    return f"sub-{str(index).zfill(3)}"


# 추출 작업 수행
def extract_subjects():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dataset, subject_list in target_subjects.items():
        for subject in subject_list:
            folder_name = generate_folder_name(subject)
            source_path = os.path.join(source_dir, folder_name)
            target_path = os.path.join(output_dir, folder_name)

            # 폴더가 존재하면 이동
            if os.path.exists(source_path):
                print(f"Moving {source_path} to {target_path}...")
                shutil.move(source_path, target_path)
            else:
                print(f"Warning: {source_path} does not exist!")


def main():
    # 실행
    extract_subjects()
    print("Data extraction completed.")


if __name__ == "__main__":
    # 원본 데이터셋 경로와 추출 대상 경로 설정
    source_dir = "/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/data/NC"  # 원본 데이터가 위치한 디렉토리
    output_dir = "/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/NC2"  # 추출된 데이터를 저장할 디렉토리

    # 데이터셋별 추출 대상 subject 정의
    target_subjects = {
        "dataset1": list(range(1, 9)),  # sub001 ~ sub008
        "dataset2": list(range(17, 42)),  # sub017 ~ sub041
        "dataset3": list(range(66, 92)),  # sub066 ~ sub091
        "dataset4": list(range(117, 125)),  # sub117 ~ sub124
        "dataset5": list(range(132, 138)),  # sub132 ~ sub137
        "dataset6": list(range(143, 159)),  # sub143 ~ sub158
    }

    main()

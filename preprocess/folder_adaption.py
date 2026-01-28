import pandas as pd
import os
import shutil

def delete_empty_dirs(path):
    if not os.path.isdir(path):
        return
    
    for subdir in os.listdir(path):
        full_subdir_path = os.path.join(path, subdir)
        delete_empty_dirs(full_subdir_path)

    if not os.listdir(path):
        os.rmdir(path)
        print(f"delete {path}")

df = pd.read_csv('vid_labels.csv')

source_dir = "/Users/yangliu/Programs/FACE/prep_image"
target_dir = "/Users/yangliu/Programs/FACE/prep_img"
os.makedirs(target_dir, exist_ok=True)

for index, row in df.iterrows():
    subject = row['Subject']
    session = row['Session']
    block = row['Block']
    start_time = row['StartT']
    end_time = row['EndT']

    source_sub_dir = os.path.join(source_dir, subject)
    ses_dir_start = f"sub-{subject}" + f"_ses_{session}"
    ses_dir = [d for d in os.listdir(source_sub_dir) if d.startswith(ses_dir_start)]

    if ses_dir:
        source_ses_dir = os.path.join(source_sub_dir, ses_dir[0])

        if os.path.exists(source_ses_dir):
            target_img_dir = os.path.join(target_dir, subject, f"ses-{session}", f"block-{block}")
            os.makedirs(target_img_dir, exist_ok=True)

            for image_file in os.listdir(source_ses_dir):
                try:
                    image_time = int(image_file.split('.')[0])
                except(IndexError, ValueError):
                    continue

                if start_time <= image_time <= end_time:
                    image_path = os.path.join(source_ses_dir, image_file)
                    target_path = os.path.join(target_img_dir, image_file)

                    shutil.copy(image_path, target_path)
                    print(f"copy {image_path} -> {target_path}")
            
            delete_empty_dirs(target_img_dir)

print("done")
        





import os
import shutil
from tqdm import tqdm


# data_root = '/mnt/nas2/Datasets_modified/action_paper_dataset/action_close'
# copy_path = 'dataset/close'
# dir_take_key = '2021'

# date_dir_data = sorted(os.listdir(data_root))
# for date_dir in tqdm(date_dir_data):
#     if dir_take_key in date_dir:
#         trimmed_dir_path = os.path.join(data_root,date_dir,'trimmed')
#         class_dir_list = sorted(os.listdir(trimmed_dir_path))

#         for class_dir in class_dir_list:
#             try:
#                 copy_class_dir_path = os.path.join(copy_path,class_dir)
#                 if not os.path.isdir(copy_class_dir_path):
#                     os.makedirs(copy_class_dir_path)

#                 class_dir_path = os.path.join(trimmed_dir_path,class_dir)
#                 class_data_list = sorted(os.listdir(class_dir_path))

#                 for data_name in class_data_list:
#                     shutil.copy(os.path.join(class_dir_path,data_name),os.path.join(copy_class_dir_path,data_name))
#             except Exception as e:
#                 print(e)


import random

left_data_number = 280
target_dir = "dataset/deep_visions_test_2nd/test"
class_list = os.listdir(target_dir)

for class_name in class_list:
    class_path = os.path.join(target_dir, class_name)
    video_list = os.listdir(class_path)

    if len(video_list) > left_data_number:
        remove_list = random.sample(video_list, k= len(video_list) - left_data_number)

        for remove_file in remove_list:
            os.remove(os.path.join(target_dir, class_name, remove_file))
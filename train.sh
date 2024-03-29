python main.py \
--train \
--cuda \
--vid_size 128 \
--train_batch_size 16 \
--num_sample_frames 8 \
--backbone_pretrained ./weights/r2plus1d_18-91a641e6.pth \
--dataset_root ~/datasets \
--dataset_name dv_yatav \
--random_shift \
--random_crop \
--random_crop_scales 0.6 1.0 \
--optimizer adam \
--base_lr 3e-4 \
--label_smooth \
--step 40 70 \
--max_epoch 90 \
--output_dir outputs
# Action Recognition Model to Monitor Illegal Dumping using Zoom-In Image Sequence Data

By Kyoungmin Ko, Hyunmin Gwak, Eunseok Lee, Gunhwi Kim, Donghyeon Moon, Youngjoo Cho and SungHwan Kim

This repository contains a Pytorch implementation of Action Recognition Model with applications to real data.

Visualization results of real data:

<img width="327" alt="스크린샷 2022-03-11 오후 5 11 14" src="https://user-images.githubusercontent.com/35245580/157828142-11765778-51b7-4b5b-9d13-1d20226800ac.png">

For more details, please refer to our paper: [Zoom-In Method](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002781652). 

### Human detect & Video crop

Zoom-In Method is explained with reference to [/data/datasets/box_detector_torch.py](https://github.com/kyoungmingo/Dump/blob/master/data/datasets/box_detector_torch.py).

<img width="633" alt="스크린샷 2022-03-11 오후 5 12 44" src="https://user-images.githubusercontent.com/35245580/157828331-bd9d380b-6e4f-4b7a-a5e0-7bc417e655ef.png">

```
def crop_point_detect(self, que, shape):
        aspect_ratio = shape[0] / shape[1]
        total_boxes = []
        for boxes in que:
            if boxes != []:
                total_boxes.extend(boxes)
        total_boxes = np.array(total_boxes)

        if len(total_boxes) > 3:
            track_x_min, track_y_min, _, _ = np.min(total_boxes, axis=0)
            _, _, track_x_max, track_y_max = np.max(total_boxes, axis=0)

            track_height = (track_y_max - track_y_min) * 1.4
            track_width = track_height / aspect_ratio

            track_x_cent = (track_x_max + track_x_min) / 2
            track_y_cent = (track_y_max + track_y_min) / 2

            track_x_min = max(0, int(track_x_cent - track_width / 2))
            track_y_min = max(0, int(track_y_cent - track_height / 2))
            track_x_max = min(608 - 1, int(track_x_cent + track_width / 2))
            track_y_max = min(608 - 1, int(track_y_cent + track_height / 2))

            return track_x_min, track_y_min, track_x_max, track_y_max
        else:
            return None, None, None, None
```

------------


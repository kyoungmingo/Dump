from .bases import DeepVisionsBaseDataset


class DeepVisionsYatav(DeepVisionsBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(dataset_dir="deepvisions_combine", *args, **kwargs)


class DeepVisionsTest(DeepVisionsBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(dataset_dir="deep_visions_test_2nd", *args, **kwargs)


class DeepVisionsJeju(DeepVisionsBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(dataset_dir="jeju_test", *args, **kwargs)

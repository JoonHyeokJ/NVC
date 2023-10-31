import os
from typing import Optional
from .imagelist2 import ImageList
from ._util import download as download_data, check_exits


class Retail(ImageList):
    image_list = {
        "train": None,
        "validation": None,
    }
    CLASSES = []

    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
        # assert task in self.image_list
        root_task = os.path.join(root, task)
        # Product.CLASSES = os.listdir(root_task)
        Retail.CLASSES = [a for a in os.listdir(root_task) if os.path.isdir(os.path.join(root_task, a))]
        Retail.CLASSES.sort()

        # breakpoint()

        # if download:
        #     list(map(lambda args: download_data(root, *args), self.download_list))
        # else:
        #     list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(Retail, self).__init__(root, Retail.CLASSES, root_task=root_task, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
        # return os.listdir(self.root)

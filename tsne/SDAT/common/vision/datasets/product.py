import os
from typing import Optional
from .imagelist2 import ImageList
from ._util import download as download_data, check_exits


class Product(ImageList):
    """`OfficeHome <http://hemanthdv.org/OfficeHome-Dataset/>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'Ar'``: Art, \
            ``'Cl'``: Clipart, ``'Pr'``: Product and ``'Rw'``: Real_World.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            Art/
                Alarm_Clock/*.jpg
                ...
            Clipart/
            Product/
            Real_World/
            image_list/
                Art.txt
                Clipart.txt
                Product.txt
                Real_World.txt
    """
    # download_list = [
    #     ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/ca3a3b6a8d554905b4cd/?dl=1"),
    #     ("Art", "Art.tgz", "https://cloud.tsinghua.edu.cn/f/4691878067d04755beab/?dl=1"),
    #     ("Clipart", "Clipart.tgz", "https://cloud.tsinghua.edu.cn/f/0d41e7da4558408ea5aa/?dl=1"),
    #     ("Product", "Product.tgz", "https://cloud.tsinghua.edu.cn/f/76186deacd7c4fa0a679/?dl=1"),
    #     ("Real_World", "Real_World.tgz", "https://cloud.tsinghua.edu.cn/f/dee961894cc64b1da1d7/?dl=1")
    # ]
    image_list = {
        "train": None,
        "validation": None,
    }
    CLASSES = []

    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
        # assert task in self.image_list
        root_task = os.path.join(root, task)
        # Product.CLASSES = os.listdir(root_task)
        Product.CLASSES = [a for a in os.listdir(root_task) if os.path.isdir(os.path.join(root_task, a))]
        Product.CLASSES.sort()

        # breakpoint()

        # if download:
        #     list(map(lambda args: download_data(root, *args), self.download_list))
        # else:
        #     list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(Product, self).__init__(root, Product.CLASSES, root_task=root_task, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
        # return os.listdir(self.root)

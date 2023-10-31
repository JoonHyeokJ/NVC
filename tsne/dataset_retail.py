import os
from typing import Optional, Callable, Tuple, Any, List
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader  # PIL
from glob import glob


class ImageList(datasets.VisionDataset):
    """A generic Dataset class for image classification

    Args:
        root (str): Root directory of dataset
        classes (list[str]): The names of all the classes
        data_list_file (str): File to read the image list from.
        transform (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line has 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride :meth:`~ImageList.parse_data_file`.
    """

    def __init__(self, root: str, classes: List[str], root_task: str,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 ret_img_path = False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        # self.samples = self.parse_data_file(data_list_file)
        self.samples = glob(root_task+'/**/*.jpg', recursive=True)
        self.classes = classes
        self.class_to_idx = {cls: idx
                             for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls
                             for idx, cls in enumerate(self.classes)}
        self.loader = default_loader
        # self.data_list_file = data_list_file
        # breakpoint()
        self.ret_img_path = ret_img_path

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index
            return (tuple): (image, target) where target is index of the target class.
        """
        # path, target = self.samples[index]
        path = self.samples[index]
        target_tmp = self.samples[index].split('/')[-2]
        target = self.class_to_idx[target_tmp]
        # print('\n')
        # print(path)
        # print(target_tmp)
        # print(target)
        # breakpoint()
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        # breakpoint()
        if not(self.ret_img_path):
            return img, target
        else:
            return img, target, path

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    @classmethod
    def domains(cls):
        """All possible domain in this dataset"""
        raise NotImplemented


class Retail(ImageList):
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

    def __init__(self, root: str, task: str, **kwargs):
        # assert task in self.image_list
        root_task = os.path.join(root, task)
        # Product.CLASSES = os.listdir(root_task)
        Retail.CLASSES = [a for a in os.listdir(root_task) if os.path.isdir(os.path.join(root_task, a))]
        Retail.CLASSES.sort()

        # breakpoint()
        super(Retail, self).__init__(root, Retail.CLASSES, root_task=root_task, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
        # return os.listdir(self.root)

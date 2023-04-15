from torchvision.datasets import VisionDataset
import pathlib
from typing import Callable, Optional, Any, Tuple
from scipy.io import loadmat

from PIL import Image


class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        classify_by_year (bool): Indicates whether/not to reduce dataset classes to simply year
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        classify_by_year: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = split
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        self._annotations = loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        
        self._classes = loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()

        if classify_by_year:
            extract_year = lambda x: int(x[-4:])
            self._classes = list(map(extract_year, self._classes))

            self._samples = [
                (str(self._images_base_path / annotation["fname"]), self._classes[annotation["class"] - 1])
                for annotation in self._annotations
            ]

        else:
            self._samples = [
                (str(self._images_base_path / annotation["fname"]), annotation["class"] - 1)
                for annotation in self._annotations
            ]

        self.class_to_idx = {cls: i for i, cls in enumerate(self._classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

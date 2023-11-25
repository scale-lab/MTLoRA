from scipy import ndimage
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import cv2
import random
import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from collections.abc import Mapping, Sequence


int_classes = int
_use_shared_memory = False


def collate_mil(batch):
    """
    Puts each data field into a tensor with outer dimension batch size.
    Custom-made for supporting MIL
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))

    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)

    elif isinstance(batch[0], str):
        return batch

    elif isinstance(batch[0], Mapping):
        batch_modified = {key: collate_mil(
            [d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        if 'edgeidx' in batch[0]:
            batch_modified['edgeidx'] = [batch[x]['edgeidx']
                                         for x in range(len(batch))]
        return batch_modified

    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate_mil(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, val_split=0.0):
        self.shuffle = shuffle
        self.dataset = dataset
        self.nbr_examples = len(dataset)
        if val_split:
            self.train_sampler, self.val_sampler = self._split_sampler(
                val_split)
        else:
            self.train_sampler, self.val_sampler = None, None

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
            'collate_fn': collate_mil
        }
        super(BaseDataLoader, self).__init__(
            sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        self.shuffle = False

        split_indx = int(self.nbr_examples * split)
        np.random.seed(0)

        indxs = np.arange(self.nbr_examples)
        np.random.shuffle(indxs)
        train_indxs = indxs[split_indx:]
        val_indxs = indxs[:split_indx]
        self.nbr_examples = len(train_indxs)

        train_sampler = SubsetRandomSampler(train_indxs)
        val_sampler = SubsetRandomSampler(val_indxs)
        return train_sampler, val_sampler

    def get_val_loader(self):
        if self.val_sampler is None:
            return None
        # self.init_kwargs['batch_size'] = 1
        return DataLoader(sampler=self.val_sampler, **self.init_kwargs)


class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(
                device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(
                device=self.device, non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break


class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, base_size=None, augment=True, val=False,
                 crop_size=321, scale=True, flip=True, rotate=False, blur=False, return_id=False):
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self.val = val
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError

    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image, label):
        if self.crop_size:
            h, w = label.shape
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(label).resize(
                (w, h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int32)

            # Center Crop
            h, w = label.shape
            start_h = (h - self.crop_size) // 2
            start_w = (w - self.crop_size) // 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]
        return image, label

    def _augmentation(self, image, label):
        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(
                    int(self.base_size*0.5), int(self.base_size*2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)
                    ) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        h, w, _ = image.shape
        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # , borderMode=cv2.BORDER_REFLECT)
            image = cv2.warpAffine(
                image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
            # ,  borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpAffine(
                label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)

        # Padding to return the correct crop size
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT, }
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)

            # Cropping
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(
                image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        return image, label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        if self.return_id:
            return self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

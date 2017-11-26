import os
from bisect import bisect

import cv2
import magic
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class VideoLoader(Dataset):
    def __init__(self, root, frame_skip=0, frame_interval=(0, 1), transform=None, **kwargs):
        assert len(frame_interval) == 2
        start, end = frame_interval
        start = max(0, start)
        end = min(1, end)
        files = sorted(map(lambda x: os.path.join(root, x), os.listdir(root)))
        files = list(filter(lambda x: magic.from_file(x, mime=True).startswith('video'), files))
        self.frame_loaders = list(map(
            lambda x: DataLoader(FrameLoader(x, frame_skip=frame_skip, start=start, end=end, transform=transform), **kwargs),
                                 files))

    def __len__(self):
        return len(self.frame_loaders)

    def __getitem__(self, index):
        return self.frame_loaders[index]


class RandomFrameLoader(Dataset):
    def __init__(self, root, frame_skip=0, frame_interval=(0, 1), transform=None):
        assert len(frame_interval) == 2
        start, end = frame_interval
        start = max(0, start)
        end = min(1, end)
        files = sorted(map(lambda x: os.path.join(root, x), os.listdir(root)))
        files = list(filter(lambda x: magic.from_file(x, mime=True).startswith('video'), files))
        self.frame_loaders = list(map(
            lambda x: FrameLoader(x, frame_skip=frame_skip, start=start, end=end, transform=transform), files))
        self.frame_counts = np.cumsum(np.array([0] + list(map(FrameLoader.__len__, self.frame_loaders))))

    def __len__(self):
        return self.frame_counts[-1]

    def __getitem__(self, index):
        video_index = bisect(self.frame_counts, index) - 1
        frame_index = index - self.frame_counts[video_index]
        return self.frame_loaders[video_index][frame_index]


class FrameLoader(Dataset):
    def __init__(self, filename, frame_skip=0, start=0, end=1, transform=None):
        self.filename = filename
        self.frame_multiplier = frame_skip + 1
        self.transform = transform
        self.capture = cv2.VideoCapture(filename)
        x, y = os.path.split(filename)[-1].split('-')[1:3]
        self.target = int(y) + (19 if x == 'M' else -1)

        self.length = int(self.capture.get(7))
        self.start = int(start * self.length)
        end = int(end * self.length)
        self.length = (end - self.start) // self.frame_multiplier

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index * self.frame_multiplier + self.start
        self.capture.set(1, index)
        frame = self.capture.read()[1]
        if self.transform is not None:
            frame = self.transform(frame)
        return frame, self.target


def train_test_data_loader(data, batch_size, test_batch_size=None,
                      test_size=0.3, **kwargs):
    if test_batch_size is None:
        test_batch_size = batch_size

    # data = RandomFrameLoader(root_dir, frame_skip=frame_skip, frame_interval=frame_interval, transform=transform)

    indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(indices, test_size=test_size)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, **kwargs)
    test_loader = DataLoader(data, batch_size=test_batch_size, sampler=test_sampler, **kwargs)

    return train_loader, test_loader


if __name__ == '__main__':
    from torchvision import transforms
    transform_list = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((240, 240)),
        transforms.Scale(128),
        transforms.Lambda(lambda x: x.convert('L')),
        transforms.ToTensor()
    ])

    data_set = RandomFrameLoader('/home/mipsa/sample', frame_skip=29, transform=transform_list)
    print("Length:", len(data_set))
    for i in range(0, len(data_set), 500):
        frame, target = data_set[i]
        cv2.imshow(str(i), frame.numpy()[0])
        # print(frame)
        # print(target)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

    data_set = VideoLoader('/home/mipsa/sample', frame_skip=14, transform=transform_list)
    print(data_set)
    print("Length:", len(data_set))
    for i in range(0, len(data_set), 100):
        frame_loader, video_target = data_set[i]
        print(i, 'Length:', len(frame_loader))
        for j in range(0, len(frame_loader), 10):
            frame, frame_target = frame_loader[j]
            cv2.imshow("%d:%d" % (i, j), frame.numpy()[0])
            print(i, j, video_target == frame_target)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

import os
from bisect import bisect

import cv2
import magic
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


class FrameLoader(Dataset):
    """Load frames from videos in given folder."""

    FRAME_INDEX_PROPERTY = 1
    FRAME_COUNT_PROPERTY = 7

    @staticmethod
    def file_mapper(filename):
        """
        Find the desired output from filename.
        XX-Y-ZZ-U.avi : Y and ZZ decide what the output is.
        Y = M has 43 values for ZZ
        Y = H has 20
        :param filename: name of the video file.
        :return: desired class
        """
        x = filename.split('-')[1:3]
        return int(x[1]) + (19 if x[0] == 'M' else -1)

    def __init__(self, root_dir, frame_skip=0, transform=None, frame_interval=(0.0, 1.0)):
        """
        Create a wrapper to easily load any frame from any video in the directory.
        :param root_dir: The directory to load videos from
        :param frame_skip: The number of frames to skip between any 2 frames
        :param transform: The transformation to be applied to frames
        :param frame_interval: The slice of the video to load frames from
        """
        self.root_dir = root_dir  # root directory for videos
        self.transform = transform  # transformation to be applied
        self.frame_multiplier = frame_skip + 1  # the index of frames to be loaded

        assert len(frame_interval) == 2
        # clamp the interval in the range 0 to 1
        if frame_interval[0] < 0:
            frame_interval = 0, frame_interval[1]
        if frame_interval[1] > 1:
            frame_interval = frame_interval[0], 1
        assert frame_interval[0] < frame_interval[1]

        # list all video file names
        files = sorted(filter(lambda x: magic.from_file(os.path.join(self.root_dir, x), mime=True).startswith('video'),
                              os.listdir(self.root_dir)))

        # get target vectors for the same: Requires different ways for different data-set
        self.out = np.array(list(map(self.file_mapper, files)), dtype='int64')

        # create Capture instances
        self.capture = list(map(cv2.VideoCapture,
                                map(lambda x: os.path.join(self.root_dir, x), files)))

        # count the number of frames that can be extracted
        # total frame counts
        self.frame_counts = np.array(list(map(lambda x: int(x.get(FrameLoader.FRAME_COUNT_PROPERTY)), self.capture)))
        # start and end
        self.frame_start = np.round(self.frame_counts * frame_interval[0], 0).astype('int32')
        frame_end = np.round(self.frame_counts * frame_interval[1], 0).astype('int32')
        # number of frames in the interval after skipping
        self.frame_counts = (frame_end - self.frame_start) // self.frame_multiplier

        # cumulative sum for faster access
        self.frame_counts = np.cumsum(self.frame_counts)

    def __len__(self):
        return self.frame_counts[-1]  # from cum-sum

    def __getitem__(self, index):
        # the video with the required frame
        video_index = bisect(self.frame_counts, index)
        # the index of frame in chosen video
        frame_index = (index - self.frame_counts[video_index - 1]) if video_index else index
        # the frame that needs to be loaded, in the interval specified
        frame_index = frame_index * self.frame_multiplier + self.frame_start[video_index]

        # load the frame from video
        capture = self.capture[video_index]
        capture.set(FrameLoader.FRAME_INDEX_PROPERTY, frame_index)
        video_frame = capture.read()[1]

        if self.transform:  # apply transforms if any
            video_frame = self.transform(video_frame)

        return video_frame, self.out[video_index]  # return frame and target

    # def __del__(self):
    #     for v in self.capture:
    #         v.release()


def train_test_loader(root_dir, batch_size,
                      frame_skip=0, frame_interval=(0.25, 0.75),
                      transform=None, test_batch_size=None,
                      test_size=0.3, **kwargs):
    if test_batch_size is None:
        test_batch_size = batch_size

    data = FrameLoader(root_dir, frame_skip=frame_skip, frame_interval=frame_interval, transform=transform)

    indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(indices, test_size=test_size)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, **kwargs)
    test_loader = DataLoader(data, batch_size=test_batch_size, sampler=test_sampler, **kwargs)

    return train_loader, test_loader


'''
if __name__ == '__main__':
    transform_list = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((240, 240)),
        transforms.Scale(128),
        transforms.Lambda(lambda x: x.convert('L')),
        transforms.ToTensor()
    ])
    data_set = FrameLoader('../small', 239, transform=transform_list)
    print("Length:", len(data_set))
    print("Root:", data_set.root_dir)
    print("Capture:", data_set.capture)
    print("Transform", data_set.transform)
    print("Counts:", data_set.frame_counts)
    for i in range(0, len(data_set), 10):
        frame, target = data_set[i]
        cv2.imshow(str(i), frame.numpy()[0])
        # print(frame)
        # print(target)
        cv2.waitKey(50)
    cv2.destroyAllWindows()
'''
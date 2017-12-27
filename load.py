import os
from bisect import bisect

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from helper import get_target


class SignData(Dataset):
    """
    Load frames in all videos as tensors.
    """
    def __init__(self, path, frame_skip=0, frame_interval=(0, 1), transform=None, **kwargs):
        """
        Create a data-loader instance for each video so that torch's built-in features are used to load frames as batch.
        :param path: The folder to load videos from.
        :param frame_skip: The number of frames to skip between 2 consecutive frames. Reduces the amount of processing.
        :param frame_interval: The part of video to process. (start, end) are provided as ratio of length.
        :param transform: Transformations to be applied on each frame.
        :param kwargs: Additional (CUDA) arguments to be passed to DataLoader instances.
        """
        path = os.path.abspath(path)

        start, end = frame_interval
        # clamp the start and end ratios
        start = max(0, start)
        end = min(1, end)
        start = min(start, end)

        # load the file names
        files = sorted([os.path.join(path, x) for x in os.listdir(path)])

        # Create SignVideo instances and remove those which have no frames to read
        self.videos = list(filter(len, (SignVideo(x, frame_skip=frame_skip, start=start, end=end, transform=transform) for x in files)))

        # Create DataLoaders for each, with batch size = length so that all frames are loaded at once
        self.video_loaders = [DataLoader(x, batch_size=len(x), **kwargs) for x in self.videos]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        # return the first (and only) batch in video-loader
        for data in self.video_loaders[index]:
            # data: (number of frames in videos[index]) x channels x width x height
            return data, self.videos[index].target


class SignVideo(Dataset):
    """
    A data-set formed by loading frames from videos.
    """
    def __init__(self, path, frame_skip=0, start=0, end=1, transform=None, target=True):
        """
        Create a wrapper to load each frame from given video.
        :param path: The path to the video.
        :param frame_skip: The number of frames to skip between 2 consecutive frames.
        :param start: The first position in the video to be considered.
        :param end: The last position in the video to be considered.
        :param transform: Transformations to be applied on each frame.
        """
        # skips are harder to use. Change to multiplier.
        self.frame_multiplier = frame_skip + 1
        self.transform = transform

        # Create a VideoCapture instance
        self.capture = cv2.VideoCapture.__call__(path)  # Just to avoid warning

        # Get the target
        if target:
            self.target = get_target(os.path.split(path)[-1])
        else:
            self.target = None

        # count the number of frames between (start, end) with skips
        self.length = int(self.capture.get(7))
        self.start = int(start * self.length)
        end = int(end * self.length)
        self.length = (end - self.start) // self.frame_multiplier

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # find the absolute position in the video
        index = index * self.frame_multiplier + self.start
        # set the position
        self.capture.set(1, index)
        # read the frame
        frame = self.capture.read()[1]
        # apply transformations
        if self.transform is not None:
            frame = self.transform(frame)
        return frame


class SignDataRandom(Dataset):
    """
    A data set that supports random access to any frame in any video.
    Improves training of CNN.
    """
    def __init__(self, sign_data):
        """
        Initializes from an instance of SignData, to save space.
        :param sign_data: The SignData instance to retrieve frames from.
        """
        self.videos = sign_data.videos
        self.video_lengths = np.cumsum([0] + [len(x) for x in self.videos])

    def __len__(self):
        return self.video_lengths[-1]

    def __getitem__(self, index):
        video_index = bisect(self.video_lengths, index) - 1
        frame_index = index - self.video_lengths[video_index]
        video = self.videos[video_index]
        return video[frame_index], video.target

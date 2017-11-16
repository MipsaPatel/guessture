import os
from bisect import bisect

import cv2
from torch.utils.data import Dataset


class FrameLoader(Dataset):
    FRAME_INDEX_PROPERTY = 1
    FRAME_COUNT_PROPERTY = 7

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.capture = list(map(cv2.VideoCapture, map(lambda x: os.path.join(self.root_dir, x), sorted(os.listdir(self.root_dir)))))
        self.frame_counts = list(map(lambda x: int(x.get(FrameLoader.FRAME_COUNT_PROPERTY)), self.capture))
        for i in range(1, len(self.frame_counts)):
            self.frame_counts[i] += self.frame_counts[i - 1]

    def __len__(self):
        return self.frame_counts[-1]

    def __getitem__(self, index):
        video_index = bisect(self.frame_counts, index)
        frame_index = (index - self.frame_counts[video_index - 1]) if video_index else index
        capture = self.capture[video_index]
        capture.set(FrameLoader.FRAME_INDEX_PROPERTY, frame_index)
        video_frame = capture.read()[1]
        if self.transform:
            video_frame = self.transform(video_frame)
        return video_frame

    def __del__(self):
        for v in self.capture:
            v.release()

if __name__ == '__main__':
    data_set = FrameLoader('/home/tsn/Videos')
    print("Length:", len(data_set))
    print("Root:", data_set.root_dir)
    print("Capture:", data_set.capture)
    print("Transform", data_set.transform)
    print("Counts:", data_set.frame_counts)
    for i in range(1, len(data_set), 1000):
        frame = data_set[i]
        cv2.imshow(str(i), frame)
        cv2.waitKey(100)
    del data_set
    cv2.destroyAllWindows()

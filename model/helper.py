import magic


def get_target(filename):
    """
    Get the class from filename. Filename follows the format:
        <Signer>-<type>-<number>-<lighting>.<extension>
    Signer: 01 - 14 : The person in the video
    type: M (motion) or H (hand shapes)
    number: 01-43 for M and 01-20 for H
    lighting: C (contrast) or D (diffuse)
    :param filename: The name of the video file.
    :return: The target value. 0 - 19 for H and 20 - 62 for M
    """
    x, y = filename.split('-')[1:3]
    # x = type; y = number
    return int(y) + (19 if x == 'M' else -1)


actions = ['O',
           'C',
           'babyO',
           'X',
           'E',
           'A',
           '5',
           'B',
           'open-8',
           'F',
           'W',
           'Y',
           'I',
           'V',
           'R',
           'H/U',
           'L',
           '1',
           '3',
           'None',
           'up',
           'down',
           'away to target - center',
           'diagonal away to target - left',
           'left to right(probably arc)',
           'right to left(probably arc)',
           'away from signer-center to target - center',
           'center - addressee toward signer - center',
           'diagonal from signer-center to target - left',
           'diagonal from signer-center target - right',
           'diagonal addressee - center to target - left',
           'diagonal addressee - center to target - right',
           'handshape change; close to open; facing down',
           'handshape change; close to open; facing away',
           'orientation vertical to orientation horizontal',
           'alternating left - right arc swing',
           'toward signer(from unspecified start)',
           'Z - shape(zig - zag)[3 strokes – straight to right, diagonal down to left, straight to right]',
           'elbow - pivot circle in vertical plane',
           'elbow - pivot circles in vertical plane',
           'elbow - pivot orientation out to orientation in',
           'away[two - handed] to target-center',
           'toward[two - handed] signer (from unspecified start)',
           'up[two - handed]',
           'down[two - handed]',
           'to left[two - handed]',
           'to right[two - handed]',
           'away from signer-head; handshape change; close to open [two - handed]; handshape change; orientation 14',
           'toward signer - face[two - handed]',
           'left to right with handshape change open to close at signer-forehead',
           'diagonal down upper left to lower right at signer-shoulder to waist',
           'elbow-pivot orientation up to orientation down[two-handed]',
           'elbow-pivot orientation out to orientation in [lower cheek]',
           'elbow-pivot orientation out to orientation in [upper cheek]',
           'contact leftside-nose to rightside-nose',
           'contact center-forehead down to contact center-chin',
           'diagonal away from vertical basehand [fingertips oriented up]',
           'bounce from motion down to contact with vertical basehand [fingertips oriented out]',
           'motion down to contact with vertical basehand [fingertips oriented out]',
           'fist to open hand',
           'index and middle finger wiggle',
           'thumb wiggle',
           'index finger closed to open']


def get_action(out):
    """
    Map the class to its description.
    :param out: the output of the neural net, the class.
    :return: The description of the class.
    """
    return actions[out]


def is_video(path):
    """
    Check if the file in the given path is a video file.
    :param path: The path to the file.
    :return: A boolean which is true if the path is a video file.
    """
    return magic.from_file(path, mime=True).startswith('video')

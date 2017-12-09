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

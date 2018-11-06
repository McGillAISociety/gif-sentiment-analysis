import cv2


def convert_gif_to_frames(gif_file_path, num_frames_to_read=float('inf')):
    """ Adapted from: https://github.com/asharma327/Read_Gif_OpenCV_Python/blob/master/gif_to_pic.py """

    # Initialize the frame number and create empty frame list
    gif = cv2.VideoCapture(gif_file_path)
    frame_num = 0
    frame_list = []

    # Loop until there are no frames left.
    try:
        while True:
            if len(frame_list) >= num_frames_to_read:
                break
            frames_remaining, frame = gif.read()
            frame_list.append(frame)

            if not frames_remaining:
                break
            frame_num += 1
    finally:
        gif.release()

    return frame_list

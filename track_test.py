from engine.track_system import Tracking_system
import sys


def main(argv):
    track = Tracking_system()
    capture = None
    yolo_type = 'tiny'
    track_type = 0
    for i in range(1, len(argv), 2):
        if argv[i] == '--capture':
            capture = argv[i + 1]
        if argv[i] == '--yolo_type':
            yolo_type = argv[i + 1]
        if argv[i] == '--track_type':
            track_type = argv[i + 1]
    print(capture, yolo_type, track_type)
    track.run(capture=capture, yolo_type=yolo_type, track_type=track_type)


if __name__ == '__main__':
    main(sys.argv)

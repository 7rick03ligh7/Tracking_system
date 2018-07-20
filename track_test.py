from engine.track_system import Tracking_system
import os
import sys

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    track = Tracking_system()
    track.run()

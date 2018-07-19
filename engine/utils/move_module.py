import math
from time import sleep
from onvif import ONVIFCamera
from multiprocessing import Process, Pipe

# class for moving


class MoveModule:

    # waits for the flag from coords method and moves the camera if the flag is set

    def move(self, child_mpipe, child_pipe, init):

        refresh_delay = 0.2
        print("before child_mpipe send")
        ntMove = [False]
        child_mpipe.send(ntMove)
        print(" after child_mpipe send")

        while True:

            ntMove = child_mpipe.recv()
            widthSafeMin = ntMove[1]
            widthSafeMax = ntMove[2]
            heightSafeMin = ntMove[3]
            heightSafeMax = ntMove[4]
            print("recieved in move", ntMove)

            if ntMove[0]:

                coordsArr = child_pipe.recv()

                # assuming that [0] is X, [1] is Y, [2] is width of the frame, [3] is height

                x = coordsArr[0]
                y = coordsArr[1]
                width = 1280
                height = 720

                # if x-ntmove1 < 0 then X is positioned to the left of the safezone

                speedX = (round((x - ntMove[1]) / ntMove[1], 2))
                print("speedX", speedX)
                speedY = (round((y - ntMove[3]) / ntMove[3], 2))
                print("speedY", speedY)

                ntMove = [True]
                child_mpipe.send(ntMove)

                # here goes importing of ptz - media - token stuff

                token = init.token
                ptz = init.ptz

                # actual moving of the camera here

                req = {'Velocity': {'Zoom': {'space': '', 'x': '0'}, 'PanTilt': {
                    'space': '', 'y': speedY, 'x': speedX}}, 'ProfileToken': token, 'Timeout': None}
                ptz.ContinuousMove(req)

                
                while True:

                    coordsArr = child_pipe.recv()
                    x = coordsArr[0]
                    y = coordsArr[1]

                    print("coords while moving ", x, y)
                    print("safezoneminX, safezonemax x, safezoney", widthSafeMin, widthSafeMax, heightSafeMin, heightSafeMax)
                    if x > widthSafeMin and x < widthSafeMax and y > heightSafeMin and y < heightSafeMax:
                        ptz.Stop({'ProfileToken': req.ProfileToken})
                        ntMove = [False]
                        child_mpipe.send(ntMove)
                        print("movement end")
            else:

                print("no need to move yet!")
                #sleep(refresh_delay)

    # checks XY coordinates and decides if the camera needs to be moved

    def coords(self, child_pipe, init, useRightThird=False):
        
        safeZoneMarginX = 0.08
        safeZoneMarginY = 0.4
        refresh_delay = 0.2
        
        parent_cpipe, child_cpipe = Pipe()
        parent_mpipe, child_mpipe = Pipe()
        move_p = Process(target=self.move, args=(
            child_mpipe, child_cpipe, init))
        move_p.start()
        print("posle processz")
        ntMove = [False]
        
        while True:

            # recieving ntMove here informs method if the camera is moving already (because move() method starts moving
            # only if True and sets it to False
            # once the movement is completed)

            coordsArr = child_pipe.recv()
            print("polu4ayu koordinaty ", coordsArr)
            ntMove = parent_mpipe.recv()
            # assuming that [0] is X, [1] is Y, [2] is width of the frame, [3] is height

            x = coordsArr[0]
            y = coordsArr[1]
            width = 1280
            height = 720
            
            parent_cpipe.send(coordsArr)
            
            if not ntMove[0]:
                print("ifnotntmove")
                # calculates safezones for left/right thirds

                if not useRightThird:
                    widthSafeMin = round((0.33 - safeZoneMarginX) * 1280)
                    widthSafeMax = round((0.33 + safeZoneMarginX) * 1280)
                    heightSafeMin = round((0.5 - safeZoneMarginY) * 720)
                    heightSafeMax = round((0.5 + safeZoneMarginY) * 720)
                else:
                    widthSafeMin = round((0.66 - safeZoneMarginX) * 1280)
                    widthSafeMax = round((0.66 + safeZoneMarginX) * 1280)
                    heightSafeMin = round((0.5 - safeZoneMarginY) * 720)
                    heightSafeMax = round((0.5 + safeZoneMarginY) * 720)

                # decides if the camera needs to be moved, sends True to move() if the movement is required

                if x > widthSafeMin and x < widthSafeMax and y > heightSafeMin and y < heightSafeMax:

                    ntMove = [False, widthSafeMin, widthSafeMax,
                              heightSafeMin, heightSafeMax]
                    parent_mpipe.send(ntMove)
                    #sleep(refresh_delay)
                    print("coords correct")
                else:

                    ntMove = [True, widthSafeMin, widthSafeMax,
                              heightSafeMin, heightSafeMax]
                    parent_mpipe.send(ntMove)
                    #sleep(refresh_delay)
                    print("jopa")

            else:

                print("Moving already")
                #sleep(refresh_delay)

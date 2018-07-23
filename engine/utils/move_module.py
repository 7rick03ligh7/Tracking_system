import math
from time import sleep
from onvif import ONVIFCamera
from multiprocessing import Process, Pipe

# class for moving


class MoveModule:

    def coords(self, child_pipe, init):
        
        # initializes constants
        
        safeZoneMarginX = 0.08
        safeZoneMarginY = 0.35
        speedDelta = 0.2
        clampSpeeds = True
        prevSpeedX = None
        prevSpeedY = None
        
        
        # sets the height and the width based on values from init module
        
        token = init.token
        ptz = init.ptz
        if init.width:
            width = init.width
        else:
            width = 1280
        if init.height:
            height = init.height
        else:
            height = 720
        
        # gets userightthird from init module
        
        useRightThird = init.UseRightThird
        
        # calculates safezones
        
        if not useRightThird:
            widthSafeMin = round((0.33 - safeZoneMarginX) * width)
            widthSafeMax = round((0.33 + safeZoneMarginX) * width))
        else:
            widthSafeMin = round((0.66 - safeZoneMarginX) * width)
            widthSafeMax = round((0.66 + safeZoneMarginX) * width)
            
        heightSafeMin = round((0.5 - safeZoneMarginY) * height)
        heightSafeMax = round((0.5 + safeZoneMarginY) * height)
        
        # factor of the speed (b for the rightside, a for the leftside)
        
        b = 1
        a = (widthSafeMin * b) / (width - widthSafeMax)
        print (a, b)
        
        # cycling through every analyzed frame to calculate the speeds and move the camera
        
        while True:

            coordsArray = child_pipe.recv()
            print("pipe recieve successful", coordsArray)

            x = coordsArray[0]
            y = coordsArray[1]
            
            # calculating speed for X, works for left third only as of now
            
            if not useRightThird:
                if x > widthSafeMin:
                    if x < widthSafeMax:
                        # x is within the safezone
                        speedX = 0
                    else:
                        # x is  on the right side
                        speedX = b*round(((x - widthSafeMax) / (width - widthSafeMax)), 2)
                        print("speedX rightside", speedX)
                else:
                    # x is on the left side
                    speedX = a*round((x - widthSafeMin) / widthSafeMin, 2)
                    print("speedX leftside", speedX)
            else:
                if x > widthSafeMin:
                    if x < widthSafeMax:
                        # x is within the safezone
                        speedX = 0
                    else:
                        # x is  on the right side
                        speedX = a*round(((x - widthSafeMax) / (width - widthSafeMax)), 2)
                        print("speedX rightside", speedX)
                else:
                    # x is on the left side
                    speedX = b*round((x - widthSafeMin) / widthSafeMin, 2)
                    print("speedX leftside", speedX)
            
            # calculating speed for Y
            
            if y > heightSafeMin:
                if y < heightSafeMax:
                    # y is within the safezone
                    speedY = 0
                else:
                    # y is in the bottom part of the screen
                    speedY = round(((y - heightSafeMax) / (height - heightSafeMax)), 2)
                    print("speedY bottom", speedY) 
            else:
                # y is in the top part of the screen
                speedY = -1*(round((y - heightSafeMin) / heightSafeMin, 2))
                print("speedY top", speedY)
                
            # "clamping" the value of speed if the delta between previous and current speeds is more than allowed
            # this ensures somewhat smooth movement of camera when speeding up, the slowing down should work automatically
            # provided there are enough frames to analyze
            # tweak this by changing speedDelta dependping on the amount of frames analyzed per second
            # disable by setting clampSpeeds to False
            
            if clampSpeeds:
                if not prevSpeedX:
                    prevSpeedX = 0
                if abs(speedX - prevSpeedX) > speedDelta:
                    if speedX - prevSpeedX < 0:
                        speedX = prevSpeedX - speedDelta
                    else:
                        speedX = prevSpeedX + speedDelta
                
                if not prevSpeedY:
                    prevSpeedY = 0
                if abs(speedY - prevSpeedY) > speedDelta
                    if speedY - prevSpeedY < 0:
                        speedY = prevSpeedY - speedDelta
                    else:
                        speedY = prevSpeedY + speedDelta
                
                prevSpeedY = speedY
                prevSpeedX = speedX
                
            
            req = {'Velocity': {'Zoom': {'space': '', 'x': '0'}, 'PanTilt': {'space': '', 'y': speedY, 'x': speedX}}, 'ProfileToken': token, 'Timeout': None}
            ptz.ContinuousMove(req)
            

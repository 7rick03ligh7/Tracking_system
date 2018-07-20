import math
from time import sleep
from onvif import ONVIFCamera
from multiprocessing import Process, Pipe

# class for moving


class MoveModule:

    def coords(self, child_pipe, init, useRightThird=False):
        
        safeZoneMarginX = 0.08
        safeZoneMarginY = 0.4

        token = init.token
        ptz = init.ptz

        # @todo: code properly
        
        width = 1280
        height = 720
        
        # calculates safezones
        
        if not useRightThird:
            widthSafeMin = round((0.33 - safeZoneMarginX) * width)
            widthSafeMax = round((0.33 + safeZoneMarginX) * width)
            heightSafeMin = round((0.5 - safeZoneMarginY) * height)
            heightSafeMax = round((0.5 + safeZoneMarginY) * height)
        else:
            widthSafeMin = round((0.66 - safeZoneMarginX) * width)
            widthSafeMax = round((0.66 + safeZoneMarginX) * width)
            heightSafeMin = round((0.5 - safeZoneMarginY) * height)
            heightSafeMax = round((0.5 + safeZoneMarginY) * height)
        
        b = 1
        a = (widthSafeMin * b) / (width - widthSafeMax)
        print (a, b)
        
        while True:

            coordsArray = child_pipe.recv()
            print("pipe recieve successful", coordsArray)

            x = coordsArray[0]
            y = coordsArray[1]
            if x > widthSafeMin:
                if x < widthSafeMax:
                    speedX = 0
                else:
                
                    # x is to on the right side
                    
                    speedX = b*round(((x - widthSafeMax) / (width - widthSafeMax)), 2)
                    print("speedX rightside", speedX)
            else:
            
                # x is on the left side
                
                speedX = a*round((x - widthSafeMin) / widthSafeMin, 2)
                print("speedX leftside", speedX)
            
            req = {'Velocity': {'Zoom': {'space': '', 'x': '0'}, 'PanTilt': {'space': '', 'y': '0', 'x': speedX}}, 'ProfileToken': token, 'Timeout': None}
            ptz.ContinuousMove(req)
            

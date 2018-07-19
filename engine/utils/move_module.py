import math
from time import sleep
from onvif import ONVIFCamera
from multiprocessing import Process, Pipe 
from engine.utils.init_module import InitModule

# class for moving

class  MoveModule:
        
    # waits for the flag from coords method and moves the camera if the flag is set
    
    def move(child_mpipe, child_pipe):
    
        refresh_delay = 0.2
        
        ntMove = [False]
        child_mpipe.send(ntMove)
        
        while True:
        
            ntMove = child_mpipe.recv()
            
            if ntMove[0]:
                
                coordsArr = child_pipe.recv()
                
                # assuming that [0] is X, [1] is Y, [2] is width of the frame, [3] is height
            
                x = coordsArr[0]
                y = coordsArr[1]
                width = coordsArr[3]
                height = coordsArr[4]
                
                # if x-ntmove1 < 0 then X is positioned to the left of the safezone
                
                speedX = round((x-ntMove[1])/ntMove[1], 2)
                print("speedX", speedX) 
                speedY = round((y-ntMove[3])/ntMove[3], 2)
                print("speedY", speedY)
                
                ntMove = [True]
                child_mpipe.send(ntMove)
                
                #code for actual moving with settings speeds, missing due to onvif for python3 being broken
                    
                
                
            else:
            
                print("no need to move yet!")
                sleep(refresh_delay)
        
    
    
    
    # checks XY coordinates and decides if the camera needs to be moved
    
    def coords(child_pipe, useRightThird=False):
    
        safeZoneMarginX = 0.04
        safeZoneMarginY = 0.15
        refresh_delay = 0.2
        
        parent_mpipe, child_mpipe = Pipe()
        move_p = Process(target = move, args=(child_mpipe, child_pipe))
        move_p.start()
        
        
        while True:
        
            # recieving ntMove here informs method if the camera is moving already (because move() method starts moving 
            # only if True and sets it to False 
            # once the movement is completed)
            
            coordsArr = child_pipe.recv()
            ntMove = parent_mpipe.recv()
            
            # assuming that [0] is X, [1] is Y, [2] is width of the frame, [3] is height
            
            x = coordsArr[0]
            y = coordsArr[1]
            width = coordsArr[3]
            height = coordsArr[4]
            
            if not ntMove[0]:
            
                # calculates safezones for left/right thirds
                
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
                    
                # decides if the camera needs to be moved, sends True to move() if the movement is required
                
                if x > widthSafeMin and x < widthSafeMax and y > heightSafeMin and y < heightSafeMax:
                
                    ntMove = [False, widthSafeMin, widthSafeMax, heightSafeMin, heightSafeMax]
                    parent_mpipe.send(ntMove)
                    sleep(refresh_delay)
                    
                else:
                
                    ntMove = [True, widthSafeMin, widthSafeMax, heightSafeMin, heightSafeMax]
                    parent_mpipe.send(ntMove)
                    sleep(refresh_delay) 
                    
            else:
            
                print("Moving already")
                sleep(refresh_delay)
            
            
            
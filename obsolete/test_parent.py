from time import sleep
from multiprocessing import Process
import os
import signal
import atexit
import sys
import test_child

def main():
    global processs
    processs = Process(target=test_child.start, name='start', args=())
    processs.start()
    sleep(3)
    print('3 seconds passed, trying to terminate now')
    kill()
    #test_child.term()
    #print(processs.is_alive())
    
def kill():
    procID = processs.pid
    os.system('pkill -TERM -P {}'.format(procID))

if __name__ == '__main__':
    main()
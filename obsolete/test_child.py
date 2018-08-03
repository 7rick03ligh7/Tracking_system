from time import sleep
from multiprocessing import Process
import os
import signal
import atexit
import sys

def proc1():
    while True:
        print('proc1 working')
        sleep(1)

def proc2():
    while True:
        print('proc2 working')
        sleep(1)

def proc3():
    while True:
        print('proc3 working')
        sleep(1)

def start():
    global process1, process2, process3
    process1 = Process(target=proc1, name='process1', args=())
    process1.start()
    process2 = Process(target=proc2, name='process2', args=())
    process2.start()
    process3 = Process(target=proc3, name='process3', args=())
    process3.start()
    print("ayy")

def term():
    process1.terminate()
    process2.terminate()
    process3.terminate()
    print('All proc terminated')

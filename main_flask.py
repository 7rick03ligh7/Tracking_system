from flask import Flask, render_template, request
import settings
from engine.track_system import Tracking_system
import track_test
import os
from multiprocessing import Process
import sys

class Config(object):      
    def __init__(self, IP, port = 80,  login ='admin', password = 'Supervisor', right = False, on = False, zoom = 0.1):   
        self.IP = IP
        self.port = port
        self.login = login
        self.password = password
        self.right = right
        self.zoom = zoom
        self.dct =  {
            'IP': self.IP,\
            'port': self.port,\
            'login': self.login,\
            'password': self.password, \
            'right': self.right,\
            'on': on,\
            'zoom': self.zoom
        }
               
    def write_to_file(self):
        cfgPath = os.getcwd() + '/engine/utils/config.py'
        f = open(cfgPath, 'w')
        f.write(str(self.dct))
        f.close()
        print('Write to file!')
        
    def get_dct(self):
        return self.dct
    
def tracking_on(camera):
    global trackingMain
    if trackingMain:
        procID = trackingMain.pid
        os.system('pkill -TERM -P {}'.format(procID))
    conf.dct['IP'] = settings.net + '.' + str(camera)
    conf.dct['on'] = True
    conf.write_to_file()
    trackingMain = Process(target=track_test.main, name='trackingMain', args=(sys.argv))
    trackingMain.start()
    return 'OK!  def tracking_on'

def tracking_off(camera):
    conf.dct['IP'] = settings.net + '.' + str(camera)
    conf.dct['on'] = False
    conf.write_to_file()
    procID = trackingMain.pid
    os.system('pkill -TERM -P {}'.format(procID))
    return 'OK!  def tracking_off'

def set_left(camera):
    try:
        #conf = Config(IP = settings.net + '.' + camera, right = False)
        #conf.write_to_file()
        conf.dct['IP'] = settings.net + '.' + str(camera)
        conf.dct['right'] = False
    except:
        return 'ERROR! def set_left'
    return 'OK!  def set_left'

def set_right(camera):
    try:
        #conf = Config(IP = settings.net + '.' + camera, right = True, zoom = zoom)
        #conf.write_to_file()
        conf.dct['IP'] = settings.net + '.' + str(camera)
        conf.dct['right'] = True
    except:
        return 'ERROR! def right'
    return 'OK!  def right'

def set_zoom(camera, zoom):
    try:
        try:
            zoom = float(zoom)
            if (zoom > 1) or (zoom < 0):
                zoom = 0
                return 'Not in interv, zoom set to 0'
        except:
            return 'Not float'
        #conf = Config(IP = settings.net + '.' + camera, zoom = zoom)        
        #conf.write_to_file()
        conf.dct['IP'] = settings.net + '.' + str(camera)
        conf.dct['zoom'] = zoom
    except:
        return 'ERROR! def zoom'
    return 'OK!  def zoom ', zoom
    
def crDict():
    global conf
    conf = Config(IP = None)
    print('config dict created')


app = Flask(__name__)
@app.route('/')
def homepage(): 
    crDict()
    return render_template('index.html', network = settings.net, cameras = settings.cameras, N = len(settings.cameras))

@app.route('/set_on')
def set_on():
    print(tracking_on(request.args['camera']))
    print('hi', request.args['camera'])
    return ' '

@app.route('/set_off')
def set_off():
    print(tracking_off(request.args['camera']))
    print('hi', request.args['camera'])
    return ' '

@app.route('/set_left')
def set_left_():
    print(set_left(request.args['camera']))
    print('hi', request.args['camera'])
    return ' '

@app.route('/set_right')
def set_right_():
    print(set_right(request.args['camera']))
    print('hi', request.args['camera'])
    return ' '

@app.route('/set_zoom')
def set_zoom_():
    print(set_zoom(request.args['camera'],request.args['zoom']))
    print('hi', request.args['camera'])
    return ' '
        
if __name__ == '__main__':
    app.run(host='192.168.11.211')
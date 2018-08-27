from flask import Flask, render_template, request
from engine.track_system import Tracking_system
import track_test
import os
from multiprocessing import Process
import sys
from onvif import ONVIFCamera
import zeep
from WSDiscovery import WSDiscovery

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
    # starts tracking
    import settings
    global trackingMain
    cameraList = settings.cameras
    # searches for the index of the selected camera
    for i in range(len(cameraList)):
        if camera == cameraList[i]:
            index = i
    # gets the port for the found index to save into config
    portList = settings.ports
    port = portList[index]
    conf.dct['IP'] = settings.net + '.' + str(camera)
    conf.dct['port'] = portList[index]
    conf.dct['on'] = True
    conf.write_to_file()
    trackingMain = Process(target=track_test.main, name='trackingMain', args=(sys.argv))
    trackingMain.start()
    print('TrackingMain PID = ', trackingMain.pid)
    return 'OK!  def tracking_on'

def tracking_off(camera):
    try:
        import settings
        conf.dct['IP'] = settings.net + '.' + str(camera)
        conf.dct['on'] = False
        conf.write_to_file()
        import pids
        listOfProc = pids.pIDs
        for i in range(len(listOfProc)):
            os.system('pkill -TERM -P {}'.format(listOfProc[i]))
            print("killed proccess pid = ", listOfProc[i])
        procID = trackingMain.pid
        #procgID = os.getpid(trackingMain)
        os.system('pkill -TERM -P {}'.format(procID))
        trackingMain.terminate()
        print("killed pid = ", procID)
        path = os.getcwd() + '/pids.py'
        pidF = open(path, 'w')
        pidF.write('pIDs = []')
        pidF.close()
    except:
        return 'unable to kill - no proccess'
    return 'OK!  def tracking_off'

def set_left(camera):
    import settings
    try:
        #conf = Config(IP = settings.net + '.' + camera, right = False)
        #conf.write_to_file()
        conf.dct['IP'] = settings.net + '.' + str(camera)
        conf.dct['right'] = False
    except:
        return 'ERROR! def set_left'
    return 'OK!  def set_left'

def set_right(camera):
    import settings
    try:
        #conf = Config(IP = settings.net + '.' + camera, right = True, zoom = zoom)
        #conf.write_to_file()
        conf.dct['IP'] = settings.net + '.' + str(camera)
        conf.dct['right'] = True
    except:
        return 'ERROR! def right'
    return 'OK!  def right'

def set_zoom(camera, zoom):
    import settings
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
    
def connect(camera):
    # connects to camera after pressing the connect button
    import settings
    global token, ptz
    zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue
    ip = settings.net + '.' + str(camera)
    cameraList = settings.cameras
    for i in range(len(cameraList)):
        if camera == cameraList[i]:
            index = i
    portList = settings.ports
    port = portList[index]
    login = 'admin'
    password = 'Supervisor'
    cam = ONVIFCamera(ip, port, login, password)
    media = cam.create_media_service()
    profile = media.GetProfiles()[0]
    token = profile.token
    ptz = cam.create_ptz_service()
    print('connected')
    
    
def increase(camera):
    # increase zoom level method
    try:
        req = {'Velocity': {'Zoom': {'space': '', 'x': '0.4'}, 'PanTilt': {'space': '', 'y': 0, 'x': 0}}, 'ProfileToken': token, 'Timeout': None}    
        ptz.ContinuousMove(req)
    except:
        return 'error def increase'
    return 'increasing'
    

def decrease(camera):
    # decrease zoom level method
    try:
        req = {'Velocity': {'Zoom': {'space': '', 'x': '-0.4'}, 'PanTilt': {'space': '', 'y': 0, 'x': 0}}, 'ProfileToken': token, 'Timeout': None}    
        ptz.ContinuousMove(req)
    except:
        return 'error def decrease'
    return 'decreasing'

def zstop(camera):
    # stops zoom when the button is released
    try:
        ptz.Stop({'ProfileToken': token})
    except:
        return 'error def zstop'
    return 'stopped'    


def crDict():
    global conf
    conf = Config(IP = None)
    print('config dict created')
    

def zeep_pythonvalue(self, xmlvalue):
    return xmlvalue


def discover():
    # discovery method
    try:
        cameras = []
        ports = []
        wsd = WSDiscovery()
        wsd.start()
        ret = wsd.searchServices()
        print('discovered ', len(ret), ' services!')
        for service in ret:
            if 'onvif' in service.getXAddrs()[0]:
                ip = service.getXAddrs()[0]
                print(ip)
                ip = ip.split('.')
                ip = ip[len(ip)-1].split(':')
                ip = ip[0].split('/')
                cameras.append(ip[0])
                port = service.getXAddrs()[0]
                if port.count(':') > 1:
                    port = port.split('.')
                    print(port)
                    port = port[len(port)-1].split(':')
                    print(port)
                    port = port[len(port)-1].split('/')
                    print(port)
                    port = port[0]
                    print(port)
                else:
                    port = 80
                ports.append(port)
                network = service.getXAddrs()[0]
                network = network.split('/')
                network = network[2].split('.')
                network = network[0] + '.' + network[1] + '.' + network[2]
        cameras.sort()
        print("cameras: ", cameras)
        print("network:" , network)
        set = open(os.getcwd() + '/settings.py', 'w')
        set.write('net = ' + repr(network) + '\n' + 'cameras = ' + repr(cameras) + '\n' + 'ports = ' + repr(ports))
        set.close()
        wsd.stop()
    except:
        return 'failed to discover services'
    return 'succ'

app = Flask(__name__)
@app.before_first_request
def startup():
    global trackingMain
    trackingMain = None

@app.route('/')
def homepage(): 
    crDict()
    discover()
    import settings
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
    
@app.route('/increase')
def increase_():
    print(increase(request.args['camera']))
    print('increase', request.args['camera'])
    return ' '
    
@app.route('/decrease')
def decrease_():
    print(decrease(request.args['camera']))
    print('decrease', request.args['camera'])
    return ' '
    
@app.route('/connect')
def connect_():
    print(connect(request.args['camera']))
    print('connect', request.args['camera'])
    return ' '
    
@app.route('/zstop')
def zstop_():
    print(zstop(request.args['camera']))
    print('zstop', request.args['camera'])
    return ' '
        
if __name__ == '__main__':
    app.run(host='192.168.11.211')
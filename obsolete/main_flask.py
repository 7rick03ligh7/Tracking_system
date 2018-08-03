from flask import Flask, render_template, request
import settings
import tracking_system

class Config(object):      
    def __init__(self, IP, port = 80,  login ='admin', password = 'Supervisor', right = True, on = False, zoom = 0.5):   
        self.IP = IP
        self.port = port
        self.login = login
        self.password = password
        self.right = right
        self.dct =  {
            'IP': self.IP,\
            'port': self.port,\
            'login': self.login,\
            'password': self.password, \
            'right': self.right,\
            'on': True,\
            'zoom': zoom
        }
               
    def write_to_file(self):
        f = open('config.py', 'w')
        f.write(str(self.dct))
        f.close()
        print('Write to file!')
        
    def get_dct(self):
        return self.dct
    
def tracking_on(camera):
    conf = Config(IP = settings.net + '.' + camera, on = True)
    conf.write_to_file()
    tracking_system.run()
    return 'OK!  def tracking_on'

def tracking_off(camera):
    try:
        conf = Config(IP = net + '.' + camera, on = False)
        conf.write_to_file()
        
    except:
        return 'ERROR! def tracking_off'
    return 'OK!  def tracking_off'

def set_left(camera):
    try:
        conf = Config(IP = net + '.' + camera, right = False)
        conf.write_to_file()
        tracking_system.run()
    except:
        return 'ERROR! def set_left'
    return 'OK!  def set_left'

def set_right(camera):
    try:
        conf = Config(IP = net + '.' + camera, right = True)
        conf.write_to_file()
        tracking_system.run()
    except:
        return 'ERROR! def right'
    return 'OK!  def right'

def set_zoom(camera, zoom):
    try:
        try:
            zoom = float(zoom)
            if (zoom > 1) or (zoom < 0):
                return 'Not in interv'
        except:
            return 'Not float'
        conf = Config(IP = net + '.' + camera, zoom = zoom)        
        conf.write_to_file()
        ##DO 
        tracking_system.run()
    except:
        return 'ERROR! def zoom'
    return 'OK!  def zoom ', zoom


app = Flask(__name__)
@app.route('/')
def homepage():   
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
    app.run(host='172.18.198.31')
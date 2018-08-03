from onvif import ONVIFCamera
from time import sleep
from ast import literal_eval
import math
import zeep
import os


class InitModule:

    # bandaid fix for broken zeep module

    def zeep_pythonvalue(self, xmlvalue):
        return xmlvalue

    def __init__(self):
        
        
        # reading config
        
        path = os.getcwd() + '/engine/utils/config.py'
        try:
            d = open(path, 'r')
            print('opening config successful')
        except IOError:
            print('error while opening config, loading default config')
            d = open(os.getcwd() + '/engine/utils/defaultConfig.py', 'r')
        dconfig = d.read()
        config = literal_eval(dconfig)
        
        # dont touch this
        
        zeep.xsd.simple.AnySimpleType.pythonvalue = self.zeep_pythonvalue
        
        # getting ip-port-log-pass-orientation-zoom from config
        
        ip = config.get('IP')
        port = config.get('port')
        login = config.get('login')
        password = config.get('password')
        self.UseRightThird = config.get('right')
        zoom = config.get('zoom')
        
        """ if code above doesnt work this the hardcoded version
        ip = '192.168.11.43'
        port = 80
        login = 'admin'
        password = 'Supervisor'
        self.UseRightThird = False
        """
        if ip == '192.168.11.44':
            port = 8999
        
        mycam = ONVIFCamera(ip, port, login, password)
        media = mycam.create_media_service()
        #profiles = media.GetProfiles()
        #subStream = len(profiles)
        #if subStream > 1:
        #self.profile = media.GetProfiles()[subStream-1]
        #else:
        self.profile = media.GetProfiles()[0]
        
        # get height - width of the frame to use in move module
        
        self.width = self.profile.VideoEncoderConfiguration.Resolution.Width
        self.height = self.profile.VideoEncoderConfiguration.Resolution.Height
        
        self.token = self.profile.token
        self.ptz = mycam.create_ptz_service()
        request = self.ptz.create_type('GetConfigurationOptions')
        request.ConfigurationToken = self.profile.PTZConfiguration.token
        ptz_configuration_options = self.ptz.GetConfigurationOptions(request)
        request = self.ptz.create_type('ContinuousMove')
        request.ProfileToken = self.profile.token
        
        # getting current absolute pantilt coordinates
        
        currentStatus = self.ptz.GetStatus(self.token)
        absX = currentStatus.Position.PanTilt.x
        absY = currentStatus.Position.PanTilt.y
        
        # setting the specified zoom
        
        zoomRequest =  {'Position': {'Zoom': {'space': '', 'x': zoom}, 'PanTilt': {'space': '', 'y': absY, 'x': absX}}, 'ProfileToken': self.token}
        self.ptz.AbsoluteMove(zoomRequest)
        
        
        # gets url in a struct
        
        streamURIStruct = media.GetStreamUri({'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': 'UDP'}, 'ProfileToken': self.token})
        
        # the url itself
        
        self.streamURL = streamURIStruct.Uri
        print(self.streamURL)
        
        currentStatus = self.ptz.GetStatus(self.token)
        print('zoom = ', currentStatus.Position.Zoom.x)
        

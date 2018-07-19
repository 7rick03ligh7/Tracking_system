# mandatory
from onvif import ONVIFCamera
from time import sleep
import math
import zeep

class InitModule:
    def zeep_pythonvalue(self, xmlvalue):
        return xmlvalue
    def __init__(self):
        # @todo: config parsing
        zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue
        # hardcoded ip-port-log-pass
        ip = '192.168.11.43'
        port = 80
        login = 'admin'
        password = 'Supervisor'
        mycam = ONVIFCamera(ip, port, login, password) 
        media = mycam.create_media_service()
        self.profile = media.GetProfiles()[0]
        token = profile.token
        ptz = mycam.create_ptz_service()
        request = ptz.create_type('GetConfigurationOptions')
        request.ConfigurationToken = profile.PTZConfiguration.token
        ptz_configuration_options = ptz.GetConfigurationOptions(request)
        request = ptz.create_type('ContinuousMove')
        request.ProfileToken = profile.token
        self.request2 = request
        # gets url in a struct
        streamURIStruct = media.GetStreamUri({'StreamSetup':{'Stream':'RTP-Unicast','Transport':'UDP'},'ProfileToken':token}) 
        # the url itself
        self.streamURL = streamURIStruct.Uri      
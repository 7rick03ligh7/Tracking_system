# mandatory
from onvif import ONVIFCamera
from time import sleep
import math
import zeep


class InitModule:

    # bandaid fix for broken zeep module

    def zeep_pythonvalue(self, xmlvalue):
        return xmlvalue

    def __init__(self):

        # @todo: config parsing

        # dont touch this

        zeep.xsd.simple.AnySimpleType.pythonvalue = self.zeep_pythonvalue

        # hardcoded ip-port-log-pass

        ip = '192.168.11.43'
        port = 80
        login = 'admin'
        password = 'Supervisor'

        mycam = ONVIFCamera(ip, port, login, password)
        media = mycam.create_media_service()
        self.profile = media.GetProfiles()[0]
        self.token = self.profile.token
        self.ptz = mycam.create_ptz_service()
        request = self.ptz.create_type('GetConfigurationOptions')
        request.ConfigurationToken = self.profile.PTZConfiguration.token
        ptz_configuration_options = self.ptz.GetConfigurationOptions(request)
        request = self.ptz.create_type('ContinuousMove')
        request.ProfileToken = self.profile.token
        self.request2 = request
        # gets url in a struct
        streamURIStruct = media.GetStreamUri(
            {'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': 'UDP'}, 'ProfileToken': self.token})
        # the url itself
        self.streamURL = streamURIStruct.Uri

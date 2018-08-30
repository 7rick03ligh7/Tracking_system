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
        #zoom = config.get('zoom')
        
        """ if code above doesnt work this the hardcoded version
        ip = '192.168.11.43'
        port = 80
        login = 'admin'
        password = 'Supervisor'
        self.UseRightThird = False
        """
        
        mycam = ONVIFCamera(ip, port, login, password)
        media = mycam.create_media_service()
        
        profiles = media.GetProfiles()
        createP = True
        profileNo = None
        for i in range(len(profiles)):
            profile = profiles[i]
            if profile.Name == 'trackingProfile':
                createP = False
                profileNo = i
                print('Profile already exists')
        if createP:
            print('Profile not found, creating new profile')
            media.CreateProfile('trackingProfile')
            self.profile = media.GetProfiles()[len(media.GetProfiles())-1]
            
            #adding video source configuration to the profile
            vidsrc = media.GetVideoSourceConfigurations()[0]
            addVidSrc = media.create_type('AddVideoSourceConfiguration')
            addVidSrc.ProfileToken = self.profile.token
            addVidSrc.ConfigurationToken = vidsrc.token
            media.AddVideoSourceConfiguration(addVidSrc)
            print('Video source config set')
            
            #adding video analytics configuration to the profile
            try:
                vidan = media.GetVideoAnalyticsConfigurations()[0]
                addVidAn = media.create_type('AddVideoAnalyticsConfiguration')
                addVidAn.ProfileToken = self.profile.token
                addVidAn.ConfigurationToken = vidan.token
                media.AddVideoAnalyticsConfiguration(addVidAn)
                print('Video analytics config set')
            except:
                print('Couldnt set video analytics')
            
            #adding PTZ configuration to the profile
            self.ptz = mycam.create_ptz_service()
            ptzc = self.ptz.GetConfigurations()[0]
            addPTZ = media.create_type('AddPTZConfiguration')
            addPTZ.ProfileToken = self.profile.token
            addPTZ.ConfigurationToken = ptzc.token
            media.AddPTZConfiguration(addPTZ)
            print('PTZ config set')
            
            #adding video encoder configuration to the profile
            vEncoderConfig = media.GetVideoEncoderConfigurations()[len(media.GetVideoEncoderConfigurations())-1]
            options = media.GetVideoEncoderConfigurationOptions()
            try:
                if options.JPEG:
                    print('JPEG encoding exists')
                    vEncoderConfig.Encoding = 'JPEG'
                    opIndex = None
                    for count in range(len(options.JPEG.ResolutionsAvailable)):
                        if options.JPEG.ResolutionsAvailable[count].Width == 1280 and options.JPEG.ResolutionsAvailable[count].Height == 720:
                            opIndex = count
                    if opIndex == None:
                        for count in range(len(options.JPEG.ResolutionsAvailable)):
                            if options.JPEG.ResolutionsAvailable[count].Width > 700 and options.JPEG.ResolutionsAvailable[count].Width <1281:
                                opIndex = count
                                break
                    if opIndex == None:
                        opIndex = len(options.JPEG.ResolutionsAvailable)-1
                    vEncoderConfig.Resolution.Width = options.JPEG.ResolutionsAvailable[opIndex].Width
                    vEncoderConfig.Resolution.Height = options.JPEG.ResolutionsAvailable[opIndex].Height
                    print('Width = ', vEncoderConfig.Resolution.Width, '; Height = ', vEncoderConfig.Resolution.Height)
                    vEncoderConfig.RateControl.FrameRateLimit = 4
                    vEncoderConfig.Quality = options.QualityRange.Max
                    print('after quality setting')
            except:
                print('JPEG encoding doesnt exist')
                vEncoderConfig.Resolution.Width = 1280
                vEncoderConfig.Resolution.Height = 720
                vEncoderConfig.RateControl.FrameRateLimit = 4
                vEncoderConfig.Quality = options.QualityRange.Max
            setVideoEncode = media.create_type('SetVideoEncoderConfiguration')
            print('type created')
            setVideoEncode.ForcePersistence = True
            setVideoEncode.Configuration = vEncoderConfig
            print('setvideoencode ready for setting')
            #media.SetVideoEncoderConfiguration(setVideoEncode)
            print('setvideoencoder success')
            addVideoEncode = media.create_type('AddVideoEncoderConfiguration')
            addVideoEncode.ProfileToken = self.profile.token
            addVideoEncode.ConfigurationToken = vEncoderConfig.token
            # not setting the video encoder because of broken onvif python3 implementation
            #media.AddVideoEncoderConfiguration(addVideoEncode)
            print('Video encoder set')
        else:
            self.profile = media.GetProfiles()[profileNo]
            print('Profile selected')
            print(self.profile.Name)
        
        # get height - width of the frame to use in move module
        
        self.width = self.profile.VideoEncoderConfiguration.Resolution.Width
        print('Got width = ', self.width)
        self.height = self.profile.VideoEncoderConfiguration.Resolution.Height
        print('Codec = ', self.profile.VideoEncoderConfiguration.Encoding)
        print('FPS = ', self.profile.VideoEncoderConfiguration.RateControl.FrameRateLimit)
        
        self.token = self.profile.token
        self.ptz = mycam.create_ptz_service()
        
        
        # getting current absolute pantilt coordinates
        
        #currentStatus = self.ptz.GetStatus(self.token)
        #absX = currentStatus.Position.PanTilt.x
        #absY = currentStatus.Position.PanTilt.y
        
        # setting the specified zoom
        
        #zoomRequest =  {'Position': {'Zoom': {'space': '', 'x': zoom}, 'PanTilt': {'space': '', 'y': absY, 'x': absX}}, 'ProfileToken': self.token}
        #self.ptz.AbsoluteMove(zoomRequest)
        
        
        # gets url in a struct
        
        streamURIStruct = media.GetStreamUri({'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': 'UDP'}, 'ProfileToken': self.token})
        
        # the url itself
        
        
        self.streamURL = 'rtsp://' + ip + ':554' + '/Streaming/Channels/2'
        print(self.streamURL)
        
        #currentStatus = self.ptz.GetStatus(self.token)
        #print('zoom = ', currentStatus.Position.Zoom.x)
        
    def bug(width, height):
        self.width = width
        self.height = height

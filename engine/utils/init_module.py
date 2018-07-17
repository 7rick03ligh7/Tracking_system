# обязательно
# всунуть куда-нибудь в начало
from onvif import ONVIFCamera
from time import sleep
import math


def initModule():

    # здесь где-то парсинг конфига

    # айпи-логин-пасс захардкодил пока нет инфы про работу веб штук
    mycam = ONVIFCamera('192.168.11.43', 80, 'admin', 'Supervisor')

    media = mycam.create_media_service()
    profile = media.GetProfiles()[0]
    token = profile._token

    ptz = mycam.create_ptz_service()
    request = ptz.create_type('GetConfigurationOptions')
    request.ConfigurationToken = profile.PTZConfiguration._token
    ptz_configuration_options = ptz.GetConfigurationOptions(request)

    request = ptz.create_type('ContinuousMove')
    request.ProfileToken = profile._token

    # получает юрл в структуре
    streamURIStruct = media.GetStreamUri(
        {'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': 'UDP'}, 'ProfileToken': token})

    # непосредственно сам ЮРЛ потока в переменной для всовывания в опенСВ
    streamURL = streamURIStruct.Uri

# -*- coding: utf-8 -*-
import os
import re
import wda
from airtest.utils.compat import decode_path


THISPATH = decode_path(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_IPROXY_PATH = {
    "Windows": os.path.join(THISPATH, "iproxy", "windows", "iproxy.exe"),
    "Darwin": os.path.join(THISPATH, "iproxy", "mac", "iproxy"),
}
DEBUG = True
IP_PATTERN = re.compile(r'(\d+\.){3}\d+')

# When some devices (6P/7P/8P) are in landscape mode, the desktop will also change to landscape mode,
# but the click coordinates are vertical screen coordinates and require special processing
# 部分设备（6P/7P/8P）在横屏时，桌面也会变成横屏，但是点击坐标是竖屏坐标，需要特殊处理
# 由于wda不能获取到手机型号，暂时用屏幕尺寸来识别是否是plus手机
# https://developer.apple.com/design/human-interface-guidelines/ios/visual-design/adaptivity-and-layout/
LANDSCAPE_PAD_RESOLUTION = [(1242, 2208)]


class CAP_METHOD(object):
    MINICAP = "MINICAP"
    WDACAP = "WDACAP"
    MJPEG = "MJPEG"

class MJpeg_Settings(object):
    # 是否启动mjpeg服务
    MJPEG_SERVER = False

    # 截图质量50-100
    MJPEG_SERVER_SCREENSHOT_QUALITY = 80

    # 帧率
    MJPEG_SERVER_FRAMERATE = 10

    # wda default mjpeg server port number
    DEFAULT_MJPEG_PORT = 9100


# now touch and ime only support wda
class TOUCH_METHOD(object):
    WDATOUCH = "WDATOUCH"


class IME_METHOD(object):
    WDAIME = "WDAIME"


ROTATION_MODE = {
    0: wda.PORTRAIT,
    270: wda.LANDSCAPE,
    90: wda.LANDSCAPE_RIGHT,
    180: wda.PORTRAIT_UPSIDEDOWN,
}


KEY_EVENTS = {
    "home": "home",
    "volumeup": "volumeUp",
    "volumedown": "volumeDown"
}

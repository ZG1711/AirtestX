# encoding=utf-8
import cv2
import numpy as np
from six import with_metaclass


class MetaDevice(type):

    REGISTRY = {}

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        meta.REGISTRY[name] = cls
        return cls


class Device(with_metaclass(MetaDevice, object)):
    """base class for test device"""

    def __init__(self):
        super(Device, self).__init__()

    def to_json(self):
        try:
            uuid = repr(self.uuid)
        except:
            uuid = None
        return f"<{self.__class__.__name__} {uuid}>"
    def find_multi_colors_touch(self, x1, y1, x2, y2, colorOne, colorOffsets, threshold=0.9):
        """
        根据find_multi_color的结果执行点击操作
        Params:
            param 参数与find_multi_color相同
        Returns: 
            Bool，表示是否找到颜色并点击成功
        """
        pos = self.find_multi_colors(x1, y1, x2, y2, colorOne, colorOffsets, threshold)
        if pos:
            self.touch(pos)
            return True
        return False
    def find_multi_colors(self, x1, y1, x2, y2, colorOne, colorOffsets, threshold=0.9):
        """
        多点找色，在指定区域内查找指定颜色
        Params:
            x1: 区域左上角x坐标
            y1: 区域左上角y坐标
            x2: 区域右下角x坐标
            y2: 区域右下角y坐标
            colorOne: 十六进制颜色值
            colorOffsets: 颜色偏移组
            threshold: 颜色匹配阈值，默认为0.9
        Returns:
            (x坐标, y坐标) Or None
        """
        # 处理主颜色,包含偏色
        if '-' in colorOne:
            main_color, offset = colorOne.split('-')
            main_color_range = self.convert_color(main_color, offset)
        else:
            main_color_range = self.convert_color(colorOne)
        
        screen = self.snapshot()
        
        # 在主颜色范围内查找第一个点
        lower = np.array(main_color_range[0], dtype=np.uint8)
        upper = np.array(main_color_range[1], dtype=np.uint8)
        mask = cv2.inRange(screen, lower, upper)
        # 在指定区域内查找
        mask_region = mask[y1:y2, x1:x2]
        
        # 查找所有匹配的主颜色点
        points = cv2.findNonZero(mask_region)
        if points is None:
            return None
        # 处理偏移颜色组
        offset_groups = []
        for group in colorOffsets.split(','):
            dx, dy, color = group.split('|')
            if '-' in color:
                offset_color, offset = color.split('-')
                color_range = self.convert_color(offset_color, offset)
            else:
                color_range = self.convert_color(color)
            offset_groups.append({
                'dx': int(dx),
                'dy': int(dy),
                'range': color_range
            })
        
        # 验证每个候选点
        for point in points:
            x, y = point[0]
            global_x, global_y = x + x1, y + y1
            
            matched_points = 0
            total_points = len(offset_groups) + 1  # 主点+所有偏移点
            
            # 检查主点是否匹配
            matched_points += 1
            
            # 检查偏移点是否匹配
            for group in offset_groups:
                check_x = global_x + group['dx']
                check_y = global_y + group['dy']
                
                if (check_x < 0 or check_y < 0 or 
                    check_x >= screen.shape[1] or check_y >= screen.shape[0]):
                    continue  # 超出屏幕范围不计入匹配
                    
                pixel_color = screen[check_y, check_x]
                lower = np.array(group['range'][0], dtype=np.uint8)
                upper = np.array(group['range'][1], dtype=np.uint8)
                
                if all(pixel_color >= lower) and all(pixel_color <= upper):
                    matched_points += 1
            match_ratio = matched_points / total_points
            
            if match_ratio >= threshold:
                return (global_x, global_y)
        
        return None
    def find_multi_colors_all(self, x1, y1, x2, y2, colorOne, colorOffsets, threshold=0.9):
        """
        多点找色，在指定区域内查找指定颜色
        Params:
            x1: 区域左上角x坐标
            y1: 区域左上角y坐标
            x2: 区域右下角x坐标
            y2: 区域右下角y坐标
            colorOne: 十六进制颜色值
            colorOffsets: 颜色偏移组
            threshold: 颜色匹配阈值，默认为0.9
        Returns:
            [(x坐标, y坐标)...] Or []
        """
        # 处理主颜色,包含偏色
        if '-' in colorOne:
            main_color, offset = colorOne.split('-')
            main_color_range = self.convert_color(main_color, offset)
        else:
            main_color_range = self.convert_color(colorOne)
        
        # 获取屏幕截图
        screen = self.snapshot()
        
        # 在主颜色范围内查找第一个点
        lower = np.array(main_color_range[0], dtype=np.uint8)
        upper = np.array(main_color_range[1], dtype=np.uint8)
        mask = cv2.inRange(screen, lower, upper)
        
        # 在指定区域内查找
        mask_region = mask[y1:y2, x1:x2]
        
        # 查找所有匹配的主颜色点
        points = cv2.findNonZero(mask_region)
        if points is None:
            return []
        
        # 处理偏移颜色组
        offset_groups = []
        for group in colorOffsets.split(','):
            dx, dy, color = group.split('|')
            if '-' in color:
                offset_color, offset = color.split('-')
                color_range = self.convert_color(offset_color, offset)
            else:
                color_range = self.convert_color(color)
            offset_groups.append({
                'dx': int(dx),
                'dy': int(dy),
                'range': color_range
            })
        
        # 存储所有匹配点
        matched_points = []
        
        # 验证每个候选点
        for point in points:
            x, y = point[0]
            global_x, global_y = x + x1, y + y1
            
            matched_count = 0
            total_points = len(offset_groups) + 1  # 主点+所有偏移点
            
            # 检查主点是否匹配
            matched_count += 1
            
            # 检查偏移点是否匹配
            for group in offset_groups:
                check_x = global_x + group['dx']
                check_y = global_y + group['dy']
                
                if (check_x < 0 or check_y < 0 or 
                    check_x >= screen.shape[1] or check_y >= screen.shape[0]):
                    continue  # 超出屏幕范围不计入匹配
                    
                pixel_color = screen[check_y, check_x]
                lower = np.array(group['range'][0], dtype=np.uint8)
                upper = np.array(group['range'][1], dtype=np.uint8)
                
                if all(pixel_color >= lower) and all(pixel_color <= upper):
                    matched_count += 1
            match_ratio = matched_count / total_points
            
            if match_ratio >= threshold:
                matched_points.append((global_x, global_y))
        
        return matched_points
    def find_image_in_region(self, x1, y1, x2, y2, image, threshold=0.9):
        """
        使用opencv2的模板匹配
        Params:
            x1: 模板区域左上角x坐标
            y1: 模板区域左上角y坐标
            x2: 模板区域右下角x坐标
            y2: 模板区域右下角y坐标
            image: 模板图片路径
            threshold: 匹配阈值(0-1)，默认为0.9
        Returns:
            如果找到则返回(x,y)，否则返回None
        """
        screen = self.snapshot()
        
        # 读取模板图像
        template = cv2.imread(image)
        if template is None:
            raise ValueError(f"无法读取模板图像: {image}")
        
        # 裁剪指定区域
        region = screen[y1:y2, x1:x2]
        
        # 使用模板匹配
        result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 检查匹配度
        if max_val >= threshold:
            # 计算全局坐标
            top_left = (max_loc[0] + x1, max_loc[1] + y1)
            return (top_left[0], top_left[1])
        return None
    def find_image_in_region_touch(self, x1, y1, x2, y2, image, threshold=0.9):
        template = cv2.imread(image)
        pos = self.find_image_in_region(x1, y1, x2, y2, image, threshold)
        if pos:
            center_x = pos[0] + template.shape[1] // 2
            center_y = pos[1] + template.shape[0] // 2
            self.touch((center_x, center_y))
            return True
        return False
    def convert_color(self,color,offsetColor=None):
        """转换颜色的偏色范围
        Params:
            color: 需要转换的十六进制颜色值
            offsetColor: 偏色的十六进制值
        Returns:
            [[B,G,R],[B,G,R]],第一个为颜色的下限,第二个为颜色的上限
        """
        b = int(color[0:2], 16)
        g = int(color[2:4], 16)
        r = int(color[4:6], 16)
        base_color = (b, g, r)  # OpenCV使用BGR格式
        
        if offsetColor is None:
            return [base_color, base_color]
        
        offset_b = int(offsetColor[0:2], 16)
        offset_g = int(offsetColor[2:4], 16)
        offset_r = int(offsetColor[4:6], 16)
        
        # 计算偏色范围
        lower_color = [
            max(0, b - offset_b),
            max(0, g - offset_g),
            max(0, r - offset_r)
        ]
        
        upper_color = [
            min(255, b + offset_b),
            min(255, g + offset_g),
            min(255, r + offset_r)
        ]
        
        return [lower_color, upper_color]
    @property
    def uuid(self):
        self._raise_not_implemented_error()

    def shell(self, *args, **kwargs):
        self._raise_not_implemented_error()

    def snapshot(self, *args, **kwargs):
        self._raise_not_implemented_error()

    def touch(self, target, **kwargs):
        self._raise_not_implemented_error()

    def double_click(self, target):
        raise NotImplementedError

    def swipe(self, t1, t2, **kwargs):
        self._raise_not_implemented_error()

    def keyevent(self, key, **kwargs):
        self._raise_not_implemented_error()

    def text(self, text, enter=True):
        self._raise_not_implemented_error()

    def start_app(self, package, **kwargs):
        self._raise_not_implemented_error()

    def stop_app(self, package):
        self._raise_not_implemented_error()

    def clear_app(self, package):
        self._raise_not_implemented_error()

    def list_app(self, **kwargs):
        self._raise_not_implemented_error()

    def install_app(self, uri, **kwargs):
        self._raise_not_implemented_error()

    def uninstall_app(self, package):
        self._raise_not_implemented_error()

    def get_current_resolution(self):
        self._raise_not_implemented_error()

    def get_render_resolution(self):
        self._raise_not_implemented_error()

    def get_ip_address(self):
        self._raise_not_implemented_error()

    def set_clipboard(self, text):
        self._raise_not_implemented_error()

    def get_clipboard(self):
        self._raise_not_implemented_error()

    def paste(self):
        self.text(self.get_clipboard())

    def _raise_not_implemented_error(self):
        platform = self.__class__.__name__
        raise NotImplementedError("Method not implemented on %s" % platform)

    def disconnect(self):
        pass

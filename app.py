import json
import os
import requests
import time
import base64
import io
from uuid import uuid4
from pathlib import Path
from copy import deepcopy
from threading import Thread, Lock
from datetime import datetime
from collections import deque
from agent import plan_main, modify_plan
from flask import Flask, request, jsonify
from flask_cors import CORS

# 语音识别相关导入
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: OpenAI库未安装，语音识别功能将不可用")

from logging import Logger
logger = Logger(__name__)

app = Flask(__name__)
CORS(app)  # 启用CORS支持前端跨域访问

# 语音识别配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 从环境变量获取
SPEECH_TO_TEXT_ENABLED = OPENAI_AVAILABLE and OPENAI_API_KEY

# 支持的音频格式
SUPPORTED_AUDIO_FORMATS = {
    'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'
}

# 语音识别配置
SPEECH_CONFIG = {
    'model': 'whisper-1',
    'language': 'zh',  # 中文
    'max_file_size': 25 * 1024 * 1024,  # 25MB
    'timeout': 30  # 30秒超时
}

# 速率限制器类
class RateLimiter:
    """
    简单的速率限制器，用于限制API调用频率
    """
    def __init__(self, max_calls=3, time_window=1):
        """
        初始化速率限制器
        
        Args:
            max_calls (int): 时间窗口内最大调用次数
            time_window (int): 时间窗口长度（秒）
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = Lock()
    
    def wait_if_needed(self):
        """
        如果需要，等待直到可以进行下一次调用
        """
        with self.lock:
            now = time.time()
            
            # 移除超出时间窗口的调用记录
            while self.calls and self.calls[0] <= now - self.time_window:
                self.calls.popleft()
            
            # 如果当前调用数已达上限，计算需要等待的时间
            if len(self.calls) >= self.max_calls:
                wait_time = self.calls[0] + self.time_window - now
                if wait_time > 0:
                    time.sleep(wait_time)
                    # 重新清理过期记录
                    now = time.time()
                    while self.calls and self.calls[0] <= now - self.time_window:
                        self.calls.popleft()
            
            # 记录本次调用
            self.calls.append(now)
            
    def get_stats(self):
        """
        获取当前速率限制器状态
        """
        with self.lock:
            now = time.time()
            # 清理过期记录
            while self.calls and self.calls[0] <= now - self.time_window:
                self.calls.popleft()
            
            return {
                "current_calls": len(self.calls),
                "max_calls": self.max_calls,
                "time_window": self.time_window,
                "next_reset": self.calls[0] + self.time_window if self.calls else now
            }

# 创建全局速率限制器：3次/秒
geocoding_rate_limiter = RateLimiter(max_calls=3, time_window=1)

# 地理编码配置
AMAP_KEY = "27c570e372bc4b6e2fcb54b9c2ed4212"  # 高德地图API密钥
GEOCODING_DB_PATH = Path(__file__).parent / "geocoding_database.json"

# 加载地理编码数据库
def load_geocoding_database():
    """加载地理编码数据库"""
    if GEOCODING_DB_PATH.exists():
        try:
            with open(GEOCODING_DB_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载地理编码数据库失败: {e}")
            return {"cache": {}, "last_updated": None, "version": "1.0"}
    else:
        return {"cache": {}, "last_updated": None, "version": "1.0"}

# 保存地理编码数据库
def save_geocoding_database(db_data):
    """保存地理编码数据库"""
    try:
        db_data["last_updated"] = datetime.now().isoformat()
        with open(GEOCODING_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, ensure_ascii=False, indent=2)
        logger.info("地理编码数据库保存成功")
        return True
    except Exception as e:
        logger.error(f"保存地理编码数据库失败: {e}")
        return False

# 初始化地理编码数据库
geocoding_db = load_geocoding_database()

def geocode_with_amap(address, city=None):
    """
    使用高德地图搜索API查找地点信息
    
    Args:
        address (str): 要搜索的地点名称或关键词
        city (str, optional): 指定查询的城市，可以是中文城市名、拼音、citycode或adcode
    
    Returns:
        dict: 包含经纬度和详细地址信息的字典，失败时返回None
    """
    if not address or not address.strip():
        return None
    
    try:
        # 应用速率限制
        geocoding_rate_limiter.wait_if_needed()
        
        # 高德地图搜索API - 使用GET请求
        url = "https://restapi.amap.com/v3/place/text"
        params = {
            'key': AMAP_KEY,
            'keywords': address.strip(),
            'output': 'json',  # 返回JSON格式
            'page': '1',       # 返回第1页
            'offset': '20'     # 每页返回20条记录
        }
        
        # 如果指定了城市，添加city参数
        if city and city.strip():
            params['city'] = city.strip()
        
        # 发送GET请求
        response = requests.get(url, params=params, timeout=10)
        logger.info(f"搜索请求: {response.url}")
        logger.info(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            
            # 检查API响应状态
            if data.get('status') == '1':
                pois = data.get('pois', [])
                if pois and len(pois) > 0:
                    poi = pois[0]  # 取第一个搜索结果
                    
                    # 解析坐标点 - location格式为"经度,纬度"
                    location = poi.get('location', '')
                    if location:
                        coords = location.split(',')
                        if len(coords) == 2:
                            try:
                                longitude = float(coords[0])
                                latitude = float(coords[1])
                                
                                # 构建完整的地址信息
                                name = poi.get('name', address)
                                address_detail = poi.get('address', '')
                                formatted_address = f"{address_detail}{name}" if address_detail else name
                                
                                # 返回详细信息
                                result = {
                                    'longitude': longitude,
                                    'latitude': latitude,
                                    'name': name,
                                    'formatted_address': formatted_address,
                                    'address': address_detail,
                                    'pname': poi.get('pname', ''),      # 省份名称
                                    'cityname': poi.get('cityname', ''), # 城市名称
                                    'adname': poi.get('adname', ''),     # 区域名称
                                    'type': poi.get('type', ''),         # POI类型
                                    'typecode': poi.get('typecode', ''), # POI类型编码
                                    'tel': poi.get('tel', ''),           # 电话
                                    'adcode': poi.get('adcode', ''),     # 区域编码
                                    'citycode': poi.get('citycode', ''), # 城市编码
                                    'business_area': poi.get('business_area', ''), # 商圈
                                    'alias': poi.get('alias', ''),       # 别名
                                    'tag': poi.get('tag', ''),           # 标签
                                    'distance': poi.get('distance', ''), # 距离（如果有参考点）
                                    'direction': poi.get('direction', ''), # 方向（如果有参考点）
                                    'level': 'POI'  # 标识这是POI搜索结果
                                }
                                
                                return result
                                
                            except (ValueError, TypeError) as e:
                                logger.warning(f"坐标解析错误: {e}")
                                pass
                else:
                    logger.info("未找到搜索结果")
                    pass
            else:
                # API返回错误
                error_info = data.get('info', '未知错误')
                infocode = data.get('infocode', '')
                logger.error(f"API错误: {error_info} (代码: {infocode})")
        else:
            logger.error(f"HTTP请求失败: {response.status_code}")
                
        return None
        
    except requests.exceptions.Timeout:
        logger.warning("请求超时")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning("连接错误")
        return None
    except Exception as e:
        logger.error(f"未知错误: {e}")
        return None

def get_geocoding_from_cache_or_api(address, city=None):
    """
    从缓存或API获取地理编码结果
    
    Args:
        address (str): 地址信息
        city (str, optional): 指定查询的城市
    
    Returns:
        dict: 地理编码结果，失败时返回None
    """
    if not address or not address.strip():
        return None
    
    # 构建缓存键，包含城市信息
    address_key = address.strip()
    if city and city.strip():
        cache_key = f"{address_key}@{city.strip()}"
    else:
        cache_key = address_key
    
    # 先检查缓存
    if cache_key in geocoding_db['cache']:
        return geocoding_db['cache'][cache_key]
    
    # 缓存中没有，调用API
    result = geocode_with_amap(address_key, city)
    
    if result:
        # 保存到缓存
        geocoding_db['cache'][cache_key] = result
        save_geocoding_database(geocoding_db)
        return result
    
    return None
class Task:
    def __init__(self, task_id:str, request:dict):
        self.request = request
        self.status = 'init'
        self.thread = None
        self.result = None
        self.id = task_id
    
    def plan(self):
        def run_plan():
            nonlocal self
            result = plan_main(self.request, self.id)
            self.result = json.loads(Path(f"query_results/{self.id}/plan.json").read_text()),
            self.status = 'completed'

        self.status = 'running'
        self.thread = Thread(target=run_plan)
        self.thread.start()
    
    def get_result(self):
        if self.status == 'completed':
            return True, self.result
        else:
            return False, None

tasks: dict[str, 'Task'] = {}
def load_results():
    result_root = Path(__file__).parent / "query_results"
    for dir in result_root.iterdir():
        if not dir.is_dir():
            continue

        task_id = dir.name
        request_data = (dir / "request.json").read_text()
        tasks[task_id] = Task(task_id, json.loads(request_data))
        if (dir / "plan.json").exists():
            tasks[task_id].status = 'completed'
            tasks[task_id].result = {
                "success": 1,
                "plan": json.loads((dir / "plan.json").read_text()),
                "brief_plan": json.loads((dir / "brief_plan.json").read_text())
            }
load_results()

@app.route('/', methods=['GET'])
def hello():
    return jsonify({"status": "ok", "message": "Welcome to the Travel Planning API!"})

@app.route("/plan/start", methods=["POST"])
def get_request():
    """
    接收旅行规划请求并启动异步任务

    Args:
        请求体JSON格式，包含以下必需字段：
        - daysCount (int): 旅行天数，范围建议1-7天
        - startCity (str): 出发城市，必须使用中文城市名
        - destinationCity (str): 目的地城市，必须使用中文城市名
        - peopleCount (int): 旅行人数，范围建议1-10人
        - additionalRequirements (str, optional): 额外需求描述，如"我想吃火锅"、"喜欢自然风光"等

    Returns:
        JSON响应 (HTTP 200):
        - task_id (str): 任务ID，用于后续查询结果
        - message (str): 固定为"Task is running in the background."

    Notes:
        - 任务为异步执行，需要使用返回的task_id调用get_plan接口获取结果
        - 支持的城市列表：上海、北京、深圳、广州、重庆、苏州、成都、杭州、武汉、南京
        - 任务执行时间取决于旅行天数和复杂度，通常需要几秒到几十秒
    """
    global tasks

    request_data = request.json
    task_id = str(uuid4())
    tasks[task_id] = Task(task_id, request_data)
    tasks[task_id].plan()

    return jsonify({
        "id": task_id,
        "status": "running",
    })


def modify_plan():
    """
    接收用户的修改请求, 并启动异步重新规划

    Args:
        task_id (str): 任务ID
        request (str): 修改的描述
    """


@app.route("/plan/result/<task_id>", methods=["GET"])
def get_plan(task_id):
    """
    获取旅行规划结果

    Args:
        task_id (str): 任务ID，通过URL参数传递，从test_plan接口获得
        day (int, optional): 指定获取第几天的行程，不传则返回完整行程概览

    Returns:
        JSON响应，包含以下可能的key：

        成功情况 (HTTP 200):
        - success (int): 固定为1，表示任务成功
        - plan (dict): 旅行计划详情，包含以下子key：
            - intercity_transport_start (dict): 出发城际交通信息（仅在day=1时存在）
                - start (str): 出发地点
                - end (str): 到达地点
                - start_time (str): 出发时间
                - end_time (str): 到达时间
                - cost (int): 费用
                - mode (str): 交通方式
            - intercity_transport_end (dict): 返回城际交通信息（仅在day=最后一天时存在）
                - start (str): 出发地点
                - end (str): 到达地点
                - start_time (str): 出发时间
                - end_time (str): 到达时间
                - cost (int): 费用
                - mode (str): 交通方式
            - activities (list): 当日活动列表（当指定day时）或每日活动列表（当未指定day时）
                每个活动包含：
                - position (str): 活动地点
                - type (str): 活动类型（breakfast/lunch/dinner/attraction等）
                - start_time (str): 开始时间
                - end_time (str): 结束时间
                - cost (int): 活动费用
                - picture (str): 活动图片URL
                - trans_time (int, optional): 交通时间（分钟）
                - trans_distance (float, optional): 交通距离（公里）
                - trans_cost (int, optional): 交通费用
                - trans_type (str, optional): 交通方式
                - trans_detail (list, optional): 详细交通信息
                - food_list (str, optional): 推荐菜品（仅餐厅类型活动）
                - latitude (float): 活动地点纬度
                - longitude (float): 活动地点经度
            - position_detail (list, optional): 位置坐标列表（当指定day时）
            - target_city (str, optional): 目标城市（当指定day时）

        失败情况 (HTTP 404):
        - error (str): 错误信息
            - "Task failed.": 当任务执行失败时（success=0）
            - "Task ID not found.": 当task_id不存在时
    """
    if task_id not in tasks:
        return jsonify({"status": "error", "result": "Task ID not found."}), 404
    
    status, result = tasks[task_id].get_result()
    if not status:
        return jsonify({"status": "running", "result": "Task is still running."}), 200

    day = request.args.get("day")
    day = int(day) if day and day.isdigit() else -1  # 默认值为-1，表示不指定天数

    tmp = deepcopy(result)
    
    # 为活动添加地理编码信息
    def add_geocoding_to_activities(activities, target_city=None):
        """为活动列表添加地理编码信息"""
        # 需要地理编码的活动类型
        geocoding_types = {
            'attraction',      # 景点
            'accommodation',   # 住宿
            'breakfast',       # 早餐
            'lunch',          # 午餐
            'dinner'          # 晚餐
        }
        
        if isinstance(activities, list):
            for activity in activities:
                if isinstance(activity, dict) and 'position' in activity and 'type' in activity:
                    # 只对指定类型的活动进行地理编码
                    if activity['type'] in geocoding_types:
                        position = activity['position']
                        
                        # 对于景点类型，增加更精确的搜索关键词
                        search_address = position
                        if activity['type'] == 'attraction':
                            # 为景点添加更精确的搜索词
                            if target_city and target_city not in position:
                                search_address = f"{target_city}{position}"
                            # 添加景点相关关键词提高准确性
                            if not any(keyword in position for keyword in ['景区', '公园', '博物馆', '广场', '大厦', '塔', '寺', '庙', '山', '湖', '河']):
                                search_address = f"{search_address}景点"
                        
                        # 调用地理编码获取经纬度
                        geocoding_result = get_geocoding_from_cache_or_api(search_address, target_city)
                        
                        # 如果第一次搜索失败，尝试使用原始地址
                        if not geocoding_result and search_address != position:
                            geocoding_result = get_geocoding_from_cache_or_api(position, target_city)
                        
                        if geocoding_result:
                            activity['latitude'] = geocoding_result.get('latitude')
                            activity['longitude'] = geocoding_result.get('longitude')
                            activity['formatted_address'] = geocoding_result.get('formatted_address', position)
                            
                            # 验证结果的相关性，过滤掉明显不相关的结果
                            formatted_addr = geocoding_result.get('formatted_address', '')
                            
                            # 如果返回的地址包含公交站、地铁站等交通设施，但原地址不是交通设施，则忽略
                            transport_keywords = ['公交', '地铁', '站台', '车站', '停车场', '收费站']
                            attraction_keywords = ['景区', '公园', '博物馆', '纪念馆', '广场', '大厦', '中心', '塔', '寺', '庙', '山', '湖', '河', '海', '岛']
                            
                            if activity['type'] == 'attraction':
                                # 如果原地址是景点，但返回的是交通设施，则清除地理编码结果
                                if (any(keyword in formatted_addr for keyword in transport_keywords) and 
                                    not any(keyword in position for keyword in transport_keywords) and
                                    any(keyword in position for keyword in attraction_keywords)):
                                    # 清除不准确的结果
                                    activity.pop('latitude', None)
                                    activity.pop('longitude', None)
                                    activity.pop('formatted_address', None)
    
    # 获取目标城市
    target_city = None
    if "plan" in tmp and "target_city" in tmp["plan"]:
        target_city = tmp["plan"]["target_city"]
    elif "brief_plan" in tmp and "target_city" in tmp["brief_plan"]:
        target_city = tmp["brief_plan"]["target_city"]
    
    if day != -1:
        # 查询特定天的行程
        if "plan" in tmp and "itinerary" in tmp["plan"]:
            itinerary = tmp["plan"]["itinerary"]
            if day <= len(itinerary):
                # 获取指定天的活动
                day_data = itinerary[day - 1]
                if "activities" in day_data:
                    add_geocoding_to_activities(day_data["activities"], target_city)
                
                # 构建返回结果
                result_plan = {
                    "activities": day_data["activities"],
                    "target_city": target_city
                }
                
                # 添加城际交通信息
                if day == 1 and "intercity_transport_start" in tmp["plan"]:
                    result_plan["intercity_transport_start"] = tmp["plan"]["intercity_transport_start"]
                
                if day == len(itinerary) and "intercity_transport_end" in tmp["plan"]:
                    result_plan["intercity_transport_end"] = tmp["plan"]["intercity_transport_end"]
                
                # 添加位置详情
                if "brief_plan" in tmp and "itinerary" in tmp["brief_plan"]:
                    if day <= len(tmp["brief_plan"]["itinerary"]):
                        brief_day_data = tmp["brief_plan"]["itinerary"][day - 1]
                        if "position_detail" in brief_day_data:
                            result_plan["position_detail"] = brief_day_data["position_detail"]
                
                tmp["plan"] = result_plan
        
        # 清理不需要的数据
        if "brief_plan" in tmp:
            del tmp["brief_plan"]
            
    else:
        # 查询完整行程
        if "plan" in tmp and "itinerary" in tmp["plan"]:
            itinerary = tmp["plan"]["itinerary"]
            for day_data in itinerary:
                if "activities" in day_data:
                    add_geocoding_to_activities(day_data["activities"], target_city)
        
        # 保持原有结构，只返回plan部分
        tmp = tmp["plan"] if "plan" in tmp else tmp
    
    return jsonify({"status": "success", "result": tmp}), 200

@app.route("/geocoding", methods=["POST"])
def geocoding():
    """
    地理编码API接口
    
    Args:
        请求体JSON格式，包含以下字段：
        - address (str): 需要进行地理编码的结构化地址信息
        - city (str, optional): 指定查询的城市，可以是中文城市名、拼音、citycode或adcode
        
    Returns:
        JSON响应：
        - success (bool): 是否成功
        - data (dict): 地理编码结果，包含详细的地址和坐标信息
        - message (str): 错误信息（如果失败）
    """
    try:
        data = request.get_json()
        if not data or 'address' not in data:
            return jsonify({
                "success": False,
                "message": "缺少必需的参数: address"
            }), 400
            
        address = data['address']
        city = data.get('city')  # 可选的城市参数
        
        result = get_geocoding_from_cache_or_api(address, city)
        
        if result:
            return jsonify({
                "success": True,
                "data": result
            })
        else:
            return jsonify({
                "success": False,
                "message": "无法获取该地址的地理编码信息，请检查地址格式或网络连接"
            }), 404
            
    except Exception as e:
        logger.error(f"地理编码接口处理错误: {e}")
        return jsonify({
            "success": False,
            "message": f"服务器错误: {str(e)}"
        }), 500

@app.route("/geocoding/cache/stats", methods=["GET"])
def geocoding_cache_stats():
    """
    获取地理编码缓存统计信息
    """
    try:
        cache_size = len(geocoding_db.get('cache', {}))
        last_updated = geocoding_db.get('last_updated')
        version = geocoding_db.get('version', '1.0')
        
        return jsonify({
            "success": True,
            "data": {
                "cache_size": cache_size,
                "last_updated": last_updated,
                "version": version,
                "rate_limiter": geocoding_rate_limiter.get_stats()
            }
        })
    except Exception as e:
        logger.error(f"缓存统计接口处理错误: {e}")
        return jsonify({
            "success": False,
            "message": f"服务器错误: {str(e)}"
        }), 500

@app.route("/geocoding/rate/stats", methods=["GET"])
def geocoding_rate_stats():
    """
    获取地理编码速率限制器统计信息
    """
    try:
        rate_stats = geocoding_rate_limiter.get_stats()
        return jsonify({
            "success": True,
            "data": {
                "rate_limiter": rate_stats,
                "description": f"速率限制: {rate_stats['max_calls']} 次/{rate_stats['time_window']} 秒"
            }
        })
    except Exception as e:
        logger.error(f"速率统计接口处理错误: {e}")
        return jsonify({
            "success": False,
            "message": f"服务器错误: {str(e)}"
        }), 500

@app.route("/plan/voice", methods=["POST"])
def voice_to_plan():
    """
    语音输入生成旅行计划

    Args:
        multipart/form-data:
        - audio_file (file): 音频文件，支持格式：mp3, mp4, mpeg, mpga, m4a, wav, webm
        - language (str, optional): 语言代码，默认为'zh'（中文）
        - additional_context (str, optional): 额外上下文信息，帮助更好地理解语音内容

    Returns:
        JSON响应，包含以下key：

        成功情况 (HTTP 200):
        - success (int): 固定为1，表示成功
        - task_id (str): 任务ID，用于后续查询结果
        - transcription (str): 转写后的文本内容
        - extracted_request (dict): 从语音中提取的旅行请求信息
        - message (str): 成功消息

        错误情况 (HTTP 400/500):
        - success (int): 固定为0，表示失败
        - error (str): 错误描述
        - details (str, optional): 详细错误信息
    """
    global tasks
    
    # 检查语音识别是否可用
    if not SPEECH_TO_TEXT_ENABLED:
        return jsonify({
            "success": 0,
            "error": "语音识别功能未启用",
            "details": "请检查OpenAI API key配置或安装openai库"
        }), 400
    
    try:
        # 检查是否有音频文件
        if 'audio_file' not in request.files:
            return jsonify({
                "success": 0,
                "error": "缺少音频文件",
                "details": "请上传音频文件"
            }), 400
        
        audio_file = request.files['audio_file']
        
        # 检查文件名
        if audio_file.filename == '':
            return jsonify({
                "success": 0,
                "error": "音频文件名为空"
            }), 400
        
        # 检查文件格式
        file_ext = audio_file.filename.rsplit('.', 1)[1].lower() if '.' in audio_file.filename else ''
        if file_ext not in SUPPORTED_AUDIO_FORMATS:
            return jsonify({
                "success": 0,
                "error": f"不支持的音频格式: {file_ext}",
                "details": f"支持的格式: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
            }), 400
        
        # 读取音频文件内容
        audio_content = audio_file.read()
        
        # 检查文件大小
        if len(audio_content) > SPEECH_CONFIG['max_file_size']:
            return jsonify({
                "success": 0,
                "error": "音频文件过大",
                "details": f"文件大小超过{SPEECH_CONFIG['max_file_size'] // 1024 // 1024}MB限制"
            }), 400
        
        # 创建临时文件用于OpenAI API
        temp_file = io.BytesIO(audio_content)
        temp_file.name = audio_file.filename
        
        # 获取额外参数
        language = request.form.get('language', SPEECH_CONFIG['language'])
        additional_context = request.form.get('additional_context', '')
        
        # 调用OpenAI Whisper API进行语音转文本
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        transcript = client.audio.transcriptions.create(
            model=SPEECH_CONFIG['model'],
            file=temp_file,
            language=language,
            response_format="text"
        )
        
        transcription = transcript.strip()
        
        if not transcription:
            return jsonify({
                "success": 0,
                "error": "语音转文本失败",
                "details": "无法从音频中提取文本内容"
            }), 400
        
        # 从转写文本中提取旅行请求信息
        extracted_request = extract_travel_request_from_text(transcription, additional_context)
        
        if not extracted_request:
            return jsonify({
                "success": 0,
                "error": "无法从语音中提取旅行信息",
                "details": f"转写文本: {transcription}",
                "suggestions": "请确保语音中包含出发地、目的地、旅行天数等关键信息"
            }), 400
        
        # 创建旅行规划任务
        task_id = str(uuid4())
        tasks[task_id] = Task(task_id, extracted_request)
        tasks[task_id].plan()
        
        return jsonify({
            "success": 1,
            "task_id": task_id,
            "transcription": transcription,
            "extracted_request": extracted_request,
            "message": "语音处理成功，旅行计划生成中..."
        })
        
    except Exception as e:
        logger.error(f"语音处理错误: {e}")
        return jsonify({
            "success": 0,
            "error": "语音处理失败",
            "details": str(e)
        }), 500


def extract_travel_request_from_text(transcription, additional_context=""):
    """
    从转写文本中提取旅行请求信息
    
    Args:
        transcription (str): 语音转写的文本
        additional_context (str): 额外上下文信息
        
    Returns:
        dict: 提取的旅行请求信息，包含daysCount、startCity、destinationCity、peopleCount等
        None: 如果无法提取有效信息
    """
    try:
        # 这里使用LLM来解析语音文本并提取结构化信息
        # 首先尝试简单的规则匹配，后续可以集成更复杂的NLP处理
        
        # 合并文本内容
        full_text = f"{transcription} {additional_context}".strip()
        
        # 使用LLM进行信息提取
        from llms import model_GPT_json
        llm = model_GPT_json()
        
        extraction_prompt = f"""
请从以下文本中提取旅行规划信息，并返回JSON格式：

文本内容: {full_text}

需要提取的信息：
- daysCount: 旅行天数（数字）
- startCity: 出发城市（中文城市名）
- destinationCity: 目的地城市（中文城市名）
- peopleCount: 旅行人数（数字，默认为1）
- additionalRequirements: 额外需求（字符串）

支持的城市：上海、北京、深圳、广州、重庆、苏州、成都、杭州、武汉、南京

如果无法提取必要信息，请返回空的JSON对象 {{}}

请只返回JSON格式，不要包含其他内容：
"""
        
        response = llm([{"role": "user", "content": extraction_prompt}])
        
        # 尝试解析JSON响应
        try:
            # 清理响应文本，提取JSON部分
            response_text = response.strip()
            
            # 查找JSON开始和结束位置
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx + 1]
                extracted_info = json.loads(json_str)
                
                # 验证必要字段
                required_fields = ['daysCount', 'startCity', 'destinationCity']
                if all(field in extracted_info and extracted_info[field] for field in required_fields):
                    # 设置默认值
                    if 'peopleCount' not in extracted_info:
                        extracted_info['peopleCount'] = 1
                    if 'additionalRequirements' not in extracted_info:
                        extracted_info['additionalRequirements'] = ''
                    
                    return extracted_info
                    
        except json.JSONDecodeError:
            logger.warning("提取旅行信息时JSON解析失败")
        
        # 如果LLM提取失败，尝试简单的关键词匹配
        return extract_by_keywords(full_text)
        
    except Exception as e:
        logger.error(f"提取旅行信息时出错: {e}")
        return None


def extract_by_keywords(text):
    """
    使用关键词匹配提取旅行信息（备用方案）
    """
    import re
    
    # 支持的城市列表
    cities = ['上海', '北京', '深圳', '广州', '重庆', '苏州', '成都', '杭州', '武汉', '南京']
    
    # 提取天数
    days_patterns = [
        r'(\d+)天',
        r'(\d+)日',
        r'天数.*?(\d+)',
        r'(\d+).*?天',
        r'一周', r'1周', r'7天',
        r'两周', r'2周', r'14天',
        r'三周', r'3周', r'21天'
    ]
    
    days_count = None
    for pattern in days_patterns:
        if pattern in ['一周', '1周', '7天']:
            if any(p in text for p in ['一周', '1周']):
                days_count = 7
                break
        elif pattern in ['两周', '2周', '14天']:
            if any(p in text for p in ['两周', '2周']):
                days_count = 14
                break
        elif pattern in ['三周', '3周', '21天']:
            if any(p in text for p in ['三周', '3周']):
                days_count = 21
                break
        else:
            match = re.search(pattern, text)
            if match:
                days_count = int(match.group(1))
                break
    
    # 提取城市并分析出发地和目的地
    found_cities = []
    city_positions = []
    
    for city in cities:
        if city in text:
            found_cities.append(city)
            city_positions.append((text.find(city), city))
    
    # 按出现位置排序
    city_positions.sort(key=lambda x: x[0])
    
    # 分析出发地和目的地
    start_city = None
    destination_city = None
    
    # 寻找"从...到..."或"...去..."的模式
    from_to_patterns = [
        r'从(.+?)到(.+?)[玩游旅]',
        r'从(.+?)去(.+?)[玩游旅]',
        r'从(.+?)出发.*?去(.+?)[玩游旅]',
        r'从(.+?)出发.*?到(.+?)[玩游旅]',
        r'(.+?)到(.+?)[玩游旅]',
        r'去(.+?)[玩游旅].*?从(.+?)出发',
        r'要去(.+?)[玩游旅].*?从(.+?)出发'
    ]
    
    for pattern in from_to_patterns:
        match = re.search(pattern, text)
        if match:
            if '去' in pattern and '从' in pattern and '出发' in pattern:
                # 处理"去...从...出发"的情况
                dest_str = match.group(1)
                start_str = match.group(2)
            else:
                start_str = match.group(1)
                dest_str = match.group(2)
            
            # 在匹配的字符串中找城市
            for city in cities:
                if city in start_str:
                    start_city = city
                if city in dest_str:
                    destination_city = city
            
            if start_city and destination_city:
                break
    
    # 如果没有找到明确的方向，使用出现顺序
    if not start_city or not destination_city:
        if len(city_positions) >= 2:
            start_city = city_positions[0][1]
            destination_city = city_positions[1][1]
        elif len(city_positions) == 1:
            destination_city = city_positions[0][1]
            start_city = '北京'  # 默认出发地
    
    # 提取人数
    people_patterns = [
        r'(\d+)人',
        r'(\d+)个人',
        r'人数.*?(\d+)',
        r'一个人', r'1个人',
        r'两个人', r'2个人',
        r'三个人', r'3个人',
        r'四个人', r'4个人',
        r'五个人', r'5个人',
        r'我们有(\d+)人',
        r'我们(\d+)人',
        r'我们有(\d+)个人'
    ]
    
    people_count = 1  # 默认1人
    for pattern in people_patterns:
        if pattern in ['一个人', '1个人']:
            if any(p in text for p in ['一个人', '1个人']):
                people_count = 1
                break
        elif pattern in ['两个人', '2个人']:
            if any(p in text for p in ['两个人', '2个人']):
                people_count = 2
                break
        elif pattern in ['三个人', '3个人']:
            if any(p in text for p in ['三个人', '3个人']):
                people_count = 3
                break
        elif pattern in ['四个人', '4个人']:
            if any(p in text for p in ['四个人', '4个人']):
                people_count = 4
                break
        elif pattern in ['五个人', '5个人']:
            if any(p in text for p in ['五个人', '5个人']):
                people_count = 5
                break
        else:
            match = re.search(pattern, text)
            if match:
                people_count = int(match.group(1))
                break
    
    # 构造结果
    if days_count and start_city and destination_city:
        return {
            'daysCount': days_count,
            'startCity': start_city,
            'destinationCity': destination_city,
            'peopleCount': people_count,
            'additionalRequirements': text
        }
    
    return None

@app.route("/test")
def test_page():
    """
    提供语音输入测试页面
    """
    try:
        with open('test_voice_frontend.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
        <head><title>测试页面未找到</title></head>
        <body>
            <h1>测试页面未找到</h1>
            <p>请确保 test_voice_frontend.html 文件存在</p>
        </body>
        </html>
        """, 404

# 现有的路由
if __name__ == "__main__":
    print(f"地理编码数据库已加载，缓存条目数: {len(geocoding_db.get('cache', {}))}")
    print(f"数据库最后更新时间: {geocoding_db.get('last_updated', '未知')}")
    app.run(host="0.0.0.0", port=8082)
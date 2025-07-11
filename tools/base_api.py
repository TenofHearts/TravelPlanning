import requests
import json
import pandas as pd
from typing import Callable
import math

# ****************** 第一步：在此处填入你申请的API密钥 ******************
AMAP_KEYS = [
    # "fdba1f2fcdd9369564f871a149d6aa30", # wwm
    # "76435a409d3ebf776460e55acb1a7171", # yj
    # "7d5951bd8860e044cb5967947f46388e", # hrh
    "e8641e97905135c164bb7e21f4bcfb1c",
    "29650faace73ec6d40cac2c6ed1082d4",
    "27c570e372bc4b6e2fcb54b9c2ed4212",
    "9f2676f76bfe6cec6be5a22bac9b4d8b",  # zbw
]
_CURRENT_KEY = 0
# ********************************************************************


def get_amap_key():
    if len(AMAP_KEYS) == 0:
        return None
    global _CURRENT_KEY
    key = AMAP_KEYS[_CURRENT_KEY]
    _CURRENT_KEY = (_CURRENT_KEY + 1) % len(AMAP_KEYS)
    return key


def search_location(keywords, region=None, show_fields="children", verbose=False):
    AMAP_KEY = get_amap_key()
    if AMAP_KEY is None:
        if verbose:
            print("[API_ERROR] 错误：请先在代码中填入您申请的高德API Key！")
        return None

    url = "https://restapi.amap.com/v5/place/text"

    params = {"key": AMAP_KEY, "keywords": keywords, "page_size": 25, "page_num": 1}

    if region:
        params["region"] = region
    if show_fields:
        params["show_fields"] = show_fields
    try:
        # 第三步：发起请求并接收返回数据
        response = requests.get(url, params=params)
        response.raise_for_status()  # 如果请求失败 (状态码不是200), 则会抛出异常

        # 解析返回的 JSON 数据
        result = response.json()

        if verbose:
            print(f"[LOCATION_SEARCH] 位置搜索完成: {keywords}")

        return result["pois"][0]["location"], result["pois"][0]["citycode"]

    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"[API_ERROR] HTTP 请求发生错误: {e}")
        return None
    except json.JSONDecodeError:
        if verbose:
            print("[API_ERROR] 解析返回的JSON数据失败")
        return None


def search_keywords(
    keywords, region=None, show_fields="business,indoor,photos", verbose=False
):
    """
    调用高德 Web 服务 API v5 的关键字搜索功能。
    文档: https://lbs.amap.com/api/webservice/guide/api-v2/search#text

    :param keywords: 必填参数，查询的关键字，多个关键字用"|"分割，如"美食|酒店"
    :param region: 可选参数，指定在哪个城市/区域内搜索
    :param show_fields: 可选参数，指定要返回的POI信息字段，多个字段用","分割，如"business,indoor,photos"
    :param verbose: 是否显示详细信息
    :return: 解析后的JSON数据，或者在请求失败时返回 None
    """
    AMAP_KEY = get_amap_key()
    if AMAP_KEY is None:
        if verbose:
            print("[API_ERROR] 错误：请先在代码中填入您申请的高德API Key！")
        return None

    url = "https://restapi.amap.com/v5/place/text"

    params = {"key": AMAP_KEY, "keywords": keywords, "page_size": 25, "page_num": 1}

    if region:
        params["region"] = region
    if show_fields:
        params["show_fields"] = show_fields
    try:
        # 第三步：发起请求并接收返回数据
        response = requests.get(url, params=params)
        response.raise_for_status()  # 如果请求失败 (状态码不是200), 则会抛出异常

        # 解析返回的 JSON 数据
        result = response.json()

        if verbose:
            print(
                f"[KEYWORD_SEARCH] 关键词搜索完成: {keywords} (区域: {region or '全国'})"
            )

        return result

    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"[API_ERROR] HTTP 请求发生错误: {e}")
        return None
    except json.JSONDecodeError:
        if verbose:
            print("[API_ERROR] 解析返回的JSON数据失败")
        return None


def distance(origin, destination):
    """
    根据输入的起点和终点，计算两地之间的距离
    输入：
        origin: 起点坐标，格式：经度,纬度
        destination: 终点坐标，格式：经度,纬度
    输出：
        distance: 两地之间的距离，单位：km
    """
    # 将经纬度字符串分割为经度和纬度
    lon1, lat1 = map(float, origin.split(","))
    lon2, lat2 = map(float, destination.split(","))

    # 将经纬度转换为弧度
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # 地球半径，单位为公里
    earth_radius = 6371.0

    # 使用哈弗辛公式计算距离
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c

    return distance


def search_routine(
    origin,
    destination,
    city1,
    city2,
    type,
    time=None,
    originpoi=None,
    destinationpoi=None,
    verbose=False,
):
    """调用高德API进行公交路线规划

    :param origin: 出发点坐标，格式：经度,纬度
    :param destination: 目的地坐标，格式：经度,纬度
    :param city1: 出发点所在城市编码
    :param city2: 目的地所在城市编码
    :param type: 公交换乘策略类型
    :param time: 出发时间，格式类似 "9-54"
    :param originpoi: 出发点POI ID
    :param destinationpoi: 目的地POI ID
    :param verbose: 是否显示详细信息
    :return: 解析后的路线信息、自然语言描述、时间花费和费用
    """
    AMAP_KEY = get_amap_key()
    if AMAP_KEY is None:
        if verbose:
            print("[API_ERROR] 错误：请先在代码中填入您申请的高德API Key！")
        return None

    url = "https://restapi.amap.com/v5/direction/transit/integrated"
    params = {
        "key": AMAP_KEY,
        "origin": origin,
        "destination": destination,
        "city1": city1,
        "city2": city2,
        "type": type,
        "show_fields": "cost",
    }

    if time:
        params["time"] = time
    if originpoi:
        params["originpoi"] = originpoi
    if destinationpoi:
        params["destinationpoi"] = destinationpoi

    try:
        # 发起请求并接收返回数据
        response = requests.get(url, params=params)
        response.raise_for_status()  # 如果请求失败 (状态码不是200), 则会抛出异常

        # 解析返回的 JSON 数据
        raw_data = response.json()

        # 解析路线信息
        transit_info = parse_route_result(raw_data, verbose=verbose)

        # 生成自然语言描述
        transit_desc = format_transit_info(transit_info, verbose=verbose)

        # 计算时间花费和费用
        time_cost = {}
        fee_cost = {}

        if (
            raw_data
            and "route" in raw_data
            and "transits" in raw_data["route"]
            and len(raw_data["route"]["transits"]) > 0
        ):
            # 取第一个方案的时间信息
            first_route = raw_data["route"]["transits"][0]

            if "cost" in first_route:
                # 计算路线总耗时（秒）
                duration_seconds = int(first_route["cost"].get("duration", "0"))
                time_cost["duration_seconds"] = duration_seconds
                time_cost["duration_minutes"] = duration_seconds // 60

                # 计算结束时间
                if time:
                    try:
                        # 处理输入时间格式例如 "9-54"
                        start_hour, start_minute = map(int, time.split("-"))

                        # 计算总时间分钟数
                        total_minutes = (
                            start_hour * 60 + start_minute + (duration_seconds // 60)
                        )

                        # 计算结束小时和分钟
                        end_hour = (total_minutes // 60) % 24
                        end_minute = total_minutes % 60

                        # 格式化为 "HH:MM" 形式
                        time_cost["end_time"] = f"{end_hour:02d}:{end_minute:02d}"
                    except Exception as e:
                        if verbose:
                            print(f"[ROUTE_PLANNING] 时间格式处理错误: {e}")
                        time_cost["end_time"] = ""

                # 计算费用
                transit_fee_value = first_route["cost"].get(
                    "transit_fee"
                )  # 先不设默认值，直接获取
                if transit_fee_value:  # 如果值不是空字符串''或None
                    fee_cost["transit_fee"] = float(transit_fee_value)
                else:
                    fee_cost["transit_fee"] = 0.0  # 如果是空字符串或None，则设置为0
        # 统计发现的路线数
        routes_count = 0
        if raw_data and "route" in raw_data and "transits" in raw_data["route"]:
            routes_count = len(raw_data["route"]["transits"])

        if verbose:
            print(f"[ROUTE_PLANNING] 路线规划完成: 找到 {routes_count} 条路线方案")

        return {
            "raw_data": raw_data,
            "transit_info": transit_info,
            "transit_desc": transit_desc,
            "time_cost": time_cost,
            "fee_cost": fee_cost,
            "distance": (
                raw_data["route"]["transits"][0].get("distance", 0)
                if routes_count > 0
                else 0
            ),
        }

    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"[API_ERROR] HTTP 请求发生错误: {e}")
        return None
    except json.JSONDecodeError:
        if verbose:
            print("[API_ERROR] 解析返回的JSON数据失败")
        return None


def parse_route_result(result, verbose=False):
    """从API返回结果中提取公交/地铁信息

    :param result: API返回的原始JSON数据
    :param verbose: 是否显示详细信息
    :return: 包含公交/地铁信息的列表
    """
    transit_info = []

    # 检查API返回状态
    if result["status"] != "1":
        if verbose:
            print(
                f"[API_ERROR] API请求失败，状态码：{result['status']}，消息：{result['info']}"
            )
        return transit_info

    # 检查是否有返回的路线
    if "route" not in result or "transits" not in result["route"]:
        if verbose:
            print("[ROUTE_PARSE] 未找到路线规划信息")
        return transit_info

    # 遍历所有可能的路线方案
    transits = result["route"]["transits"]
    if verbose:
        print(f"[ROUTE_PARSE] 路线解析: 找到 {len(transits)} 条路线方案")

    for i, transit in enumerate(transits):
        route_steps = []
        segments = transit.get("segments", [])

        # 遍历路线中的每个部分
        for segment in segments:
            # 只提取公交/地铁信息，忽略步行
            if "bus" in segment and "buslines" in segment["bus"]:
                buslines = segment["bus"]["buslines"]
                for busline in buslines:
                    bus_info = {
                        "name": busline.get("name", "未知线路"),
                        "type": busline.get("type", "未知类型"),
                        "departure_stop": busline.get("departure_stop", {}).get(
                            "name", "未知站点"
                        ),
                        "arrival_stop": busline.get("arrival_stop", {}).get(
                            "name", "未知站点"
                        ),
                        "distance": busline.get("distance", "0"),
                        "via_stops": busline.get("via_stops", []),
                    }
                    route_steps.append(bus_info)

        if route_steps:
            transit_info.append(
                {
                    "route_id": i,
                    "distance": transit.get("distance", "0"),
                    "steps": route_steps,
                }
            )

    return transit_info


def format_transit_info(transit_info, verbose=False):
    """将提取的公交/地铁信息转换为自然语言描述

    :param transit_info: 包含公交/地铁信息的列表
    :param verbose: 是否显示详细信息
    :return: 自然语言描述字符串，只返回第一种路线
    """
    if not transit_info:
        if verbose:
            print("[ROUTE_FORMAT] 未找到可用的公交/地铁路线")
        return "未找到可用的公交/地铁路线"

    # 只处理第一条路线方案
    if transit_info:
        route = transit_info[0]
        route_desc = []

        for step in route["steps"]:
            # 格式：从[出发站]乘坐[线路名称]，到达[到达站]
            desc = f"从{step['departure_stop']}乘坐{step['name']}，到达{step['arrival_stop']}"

            # 如果有途经站点，可以添加途经站点信息
            via_stops = step.get("via_stops", [])
            if via_stops and len(via_stops) > 0:
                via_stops_names = [stop.get("name", "未知站点") for stop in via_stops]
                desc += f"，途经{len(via_stops)}站（{', '.join(via_stops_names)}）"

            route_desc.append(desc)

        result = "\n".join(route_desc)
        if verbose:
            print(f"[ROUTE_FORMAT] 路线描述生成完成")
        return result

    if verbose:
        print("[ROUTE_FORMAT] 未找到可用的公交/地铁路线")
    return "未找到可用的公交/地铁路线"


def test_api_select(verbose=True):
    """
    测试API搜索酒店功能
    """
    if verbose:
        print("[TEST] === 测试1: 基本搜索 ===")
    # 测试基本搜索功能
    hotels = search_keywords("南京南站", "南京", verbose=verbose)
    if verbose and hotels:
        print(f"[TEST] 找到 {len(hotels.get('pois', []))} 个搜索结果")
        print("[TEST] 前3个结果信息:")
        print(hotels.get("pois", [])[:3])

    if verbose:
        print("[TEST] === 测试2: 带过滤条件搜索 ===")
    # 测试带过滤条件的搜索
    hotels = search_keywords("酒店 新街口", "南京", verbose=verbose)
    if verbose and hotels:
        print(f"[TEST] 找到 {len(hotels.get('pois', []))} 个搜索结果")
        print("[TEST] 前3个结果信息:")
        print(hotels.get("pois", [])[:3])

    return hotels


def test_api_routine_with_real_api(verbose=True):
    """使用真实API测试路径规划功能"""
    if verbose:
        print("[TEST] 开始测试路径规划功能...")

    # 北京三元桥到望京测试
    result = search_routine(
        origin="116.466485,39.995197",  # 三元桥附近的坐标
        destination="116.46424,40.020642",  # 望京附近的坐标
        city1="010",  # 北京市编码
        city2="010",  # 北京市编码
        type="0",  # 最快捷模式
        time="9-30",  # 出发时间为早上9点30分
        verbose=verbose,
    )

    if result and verbose:
        print("[TEST] === 原始数据 ===")
        raw_data_str = json.dumps(result["raw_data"], indent=2, ensure_ascii=False)
        print(raw_data_str[:500] + "..." if len(raw_data_str) > 500 else raw_data_str)

        print("[TEST] === 自然语言描述 ===")
        print(result["transit_desc"])

        print("[TEST] === 路线时间花费 ===")
        print(f"[TEST] 总耗时: {result['time_cost'].get('duration_minutes', 0)} 分钟")
        print(f"[TEST] 结束时间: {result['time_cost'].get('end_time', '')}")

        print("[TEST] === 路线费用 ===")
        print(f"[TEST] 公交费用: {result['fee_cost'].get('transit_fee', 0)} 元")

    return result


if __name__ == "__main__":
    print("[MAIN] 开始测试路线规划功能...")

    # 注释掉的测试代码已整理，现在使用新的测试方式
    # try:
    #     result = search_routine(
    #         origin="116.466485,39.995197",
    #         destination="116.46424,40.020642",
    #         city1="010",
    #         city2="010",
    #         type="0",
    #         time="9-30",
    #         verbose=True,
    #     )

    #     if result:
    #         print("[TEST] === 自然语言描述 ===")
    #         print(result["transit_desc"])

    #         print("[TEST] === 路线时间花费 ===")
    #         print(f"[TEST] 总耗时: {result['time_cost'].get('duration_minutes', 0)} 分钟")
    #         print(f"[TEST] 结束时间: {result['time_cost'].get('end_time', '')}")

    #         print("[TEST] === 路线费用 ===")
    #         print(f"[TEST] 公交费用: {result['fee_cost'].get('transit_fee', 0)} 元")
    # except Exception as e:
    #     print(f"[ERROR] 测试过程中发生错误: {e}")

    # 测试位置搜索
    location_result = search_location("上海文化广场", "上海", verbose=True)
    print(f"[MAIN] 位置搜索结果: {location_result}")

    print("[MAIN] 测试完成!")
    # 备用测试: search_location("北京大学", "北京", verbose=True)

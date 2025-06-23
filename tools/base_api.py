import requests
import json
import pandas as pd
from typing import Callable

# ****************** 第一步：在此处填入你申请的API密钥 ******************
AMAP_KEY = "fdba1f2fcdd9369564f871a149d6aa30"
# ********************************************************************
def search_keywords(keywords, region=None,show_fields="business,indoor,photos"):
    """
    调用高德 Web 服务 API v5 的关键字搜索功能。
    文档: https://lbs.amap.com/api/webservice/guide/api-v2/search#text

    :param keywords: 必填参数，查询的关键字，多个关键字用“|”分割，如“美食|酒店”
    :param region: 可选参数，指定在哪个城市/区域内搜索
    :return: 解析后的JSON数据，或者在请求失败时返回 None
    """
    if AMAP_KEY == 'YOUR_KEY':
        print("错误：请先在代码中填入您申请的高德API Key！")
        return None
    url = "https://restapi.amap.com/v3/place/text"
    
    params = {
        'key': AMAP_KEY,
        'keywords': keywords,
        'page_size':200,
    }
    
    if region:
        params['region'] = region
    if show_fields:
        params['show_fields'] = show_fields
    try:
        # 第三步：发起请求并接收返回数据
        response = requests.get(url, params=params)
        response.raise_for_status()  # 如果请求失败 (状态码不是200), 则会抛出异常

        # 解析返回的 JSON 数据
        result = response.json()
        return result

    except requests.exceptions.RequestException as e:
        print(f"HTTP 请求发生错误: {e}")
        return None
    except json.JSONDecodeError:
        print("解析返回的JSON数据失败")
        return None

def test_api_select():
    """
    测试API搜索酒店功能
    """
    print("=== 测试1: 基本搜索 ===")
    # 测试基本搜索功能
    hotels = search_keywords("酒店", "南京")
    print(f"找到 {len(hotels)} 家酒店")
    print("\n前3家酒店信息:")
    print(hotels['pois'][:3])
    
    print("\n=== 测试2: 带过滤条件搜索 ===")
    # 测试带过滤条件的搜索
    hotels = search_keywords("酒店 新街口", "南京")
    print(f"找到 {len(hotels)} 家五星级酒店")
    print("\n前3家酒店信息:")
    print(hotels['pois'][:3])
    
    return hotels

if __name__ == "__main__":
    print("开始测试API酒店搜索功能...")
    result = test_api_select()
    print("测试完成!")

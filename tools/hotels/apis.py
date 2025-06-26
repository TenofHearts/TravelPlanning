import pandas as pd
from pandas import DataFrame
from typing import Callable
from geopy.distance import geodesic
import os
from tools.base_api import search_keywords
import sys
from datetime import datetime
# class Accommodations:

#     def __init__(self, path: str = "../../database/hotels/nanjing/hotel.csv"):
#         curdir = os.path.dirname(os.path.realpath(__file__))
#         self.path = os.path.join(curdir, path)

#         self.data = pd.read_csv(self.path).dropna()[
#             ['hotelName', 'featureHotelType', 'poi_coordinate', 'miniPrice', 'miniPriceRoom', ]]
#         self.data['Latitude'] = self.data['poi_coordinate'].apply(
#             lambda x: float(x.split("'latitude': ")[
#                             1].split(",")[0].replace("'", ""))
#         )
#         self.data['Longitude'] = self.data['poi_coordinate'].apply(
#             lambda x: float(x.split("'longitude': ")[1].split(",")[
#                             0].replace("'", "").replace("}", ""))
#         )
#         self.data["Price"] = self.data["miniPrice"]
#         self.data['numBed'] = self.data['miniPriceRoom'].apply(
#             lambda x: 1 if ("大床" in x) or ("单人" in x) else 2
#         )
#         self.data = self.data.drop(["miniPrice", "miniPriceRoom"], axis=1)
#         self.data = self.data.rename(columns={"hotelName": "Name"})
#         self.data = self.data.drop(["poi_coordinate"], axis=1)
#         self.data.columns = [x.lower() for x in self.data.columns]
#         self.data = self.data.drop(["featurehoteltype"], axis=1)
#         self.key_type_tuple_list = []
#         for key in self.data.keys():
#             self.key_type_tuple_list.append((key, type(self.data[key][0])))
#         print("Attractions loaded.")

#     def keys(self):
#         return self.key_type_tuple_list

#     def select(self, key, func: Callable) -> DataFrame:
#         if key not in self.data.keys():
#             return "Key not found."
#         bool_list = [func(x) for x in self.data[key]]
#         return self.data[bool_list]

#     def nearby(self, lat: float, lon: float, topk: int = None, dist: float = 5) -> DataFrame:
#         distance = [geodesic((lat, lon), (x, y)).km for x, y in zip(
#             self.data['latitude'], self.data['longitude'])]
#         tmp = self.data.copy()
#         tmp['distance'] = distance
#         if dist is not None:
#             tmp = tmp[tmp['distance'] < dist]
#         tmp = tmp.sort_values(by=['distance'])
#         if topk is not None:
#             return tmp.head(topk)
#         return tmp

class Accommodations:

    def __init__(self, base_path: str = "../../database/hotels/"):
        curdir = os.path.dirname(os.path.realpath(__file__))
        city_list = [
            "beijing", "shanghai", "nanjing",
            "suzhou", "hangzhou", "shenzhen",
            "chengdu", "wuhan", "guangzhou",
            "chongqing"]
        data_path_list = [os.path.join(
            curdir, f"{base_path}/{city}/hotel_{city}_uniq.csv") for city in city_list]
        self.data = {}
        for i, city in enumerate(city_list):
            self.data[city] = pd.read_csv(data_path_list[i]).dropna()[
                ['hotelName', 'featureHotelType', 'poi_coordinate', 'miniPrice', 'miniPriceRoom', ]]
            self.data[city]['Latitude'] = self.data[city]['poi_coordinate'].apply(
                lambda x: float(x.split("'latitude': ")[
                                1].split(",")[0].replace("'", ""))
            )
            self.data[city]['Longitude'] = self.data[city]['poi_coordinate'].apply(
                lambda x: float(x.split("'longitude': ")[1].split(",")[
                                0].replace("'", "").replace("}", ""))
            )
            self.data[city]["Price"] = self.data[city]["miniPrice"]
            self.data[city]['numBed'] = self.data[city]['miniPriceRoom'].apply(
                lambda x: 1 if ("大床" in x) or ("单人" in x) else 2
            )
            self.data[city] = self.data[city].drop(
                ["miniPrice", "miniPriceRoom"], axis=1)
            self.data[city] = self.data[city].rename(
                columns={"hotelName": "Name"})
            self.data[city] = self.data[city].drop(["poi_coordinate"], axis=1)
            self.data[city].columns = [x.lower()
                                       for x in self.data[city].columns]
            # self.data[city] = self.data[city].drop(
            #     ["featurehoteltype"], axis=1)

        self.key_type_tuple_list = {}
        for city in city_list:
            self.key_type_tuple_list[city] = []
            for key in self.data[city].keys():
                self.key_type_tuple_list[city].append(
                    (key, type(self.data[city].iloc[0][key]))
                )
        city_cn_list = ["北京", "上海", "南京", "苏州", "杭州",
                        "深圳", "成都", "武汉", "广州", "重庆"]
        def to_float(x):
            try:
                return float(x)
            except:
                return 0.0
        for i, city in enumerate(city_list):
            self.data[city_cn_list[i]] = self.data.pop(city)
            self.key_type_tuple_list[city_cn_list[i]
                                     ] = self.key_type_tuple_list.pop(city)
            self.data[city_cn_list[i]]['price'] = self.data[city_cn_list[i]]['price'].apply(
                to_float)    
        # print("Accommodations loaded.")

    def keys(self, city):
        return self.key_type_tuple_list[city]

    def select(self,city, keywords=None, key=None, func=None):
        """
        使用API调用获取酒店列表，替代原来的数据库查询方式
        
        :param city: 城市名称
        :param keywords: 搜索关键词，由调用方构建
        :param key: 查询的关键字类型，用于过滤结果
        :param func: 筛选条件函数
        :param query: 查询参数字典，用于价格和房型过滤
        :return: 酒店信息的DataFrame
        """
        # 如果没有提供关键词，使用默认值
        if keywords is None:
        
            keywords = "酒店"
        
        print(f"搜索关键词: {keywords}")
        
        # 调用API
        result = search_keywords(keywords=keywords, region=city)
        
        if not result or 'pois' not in result or not result['pois']:
            return pd.DataFrame()  # 返回空DataFrame
        
        # 转换API返回的结果为DataFrame
        hotels_data = []
        for poi in result['pois']:
            url_list = [
                photo['url'] 
                for photo in poi['photos'] 
                if photo and isinstance(photo, dict) and 'url' in photo
            ]
            location = poi['location'].split(",")
            hotel_data = {
                'name': poi['name'],
                'id':poi['id'],
                'featurehoteltype': poi.get('type', ''),  # 类型
                'address': poi.get('address', ''),
                'rating':poi.get('business', {}).get('rating',''),
                'cost':poi.get('business', {}).get('cost',''),
                'tag':poi.get('business', {}).get('tag',''),
                'latitude': float(location[1]) if len(location) > 1 else 0.0,
                'longitude': float(location[0]) if len(location) > 0 else 0.0,
                'phone': poi.get('business', {}).get('tel', ''),
                'photos':url_list,
                'city': city
            }
            
            hotels_data.append(hotel_data)
        
        df = pd.DataFrame(hotels_data)
        print("line164",df)
        if key and func and key in df.columns:
            bool_list = [func(x) for x in df[key]]
            df = df[bool_list] 
        if not df.empty: # 确保DataFrame不为空才保存
            temp_dir = "temp_csv_files"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"hotels_{city}_{timestamp}.csv"
            file_path = os.path.join(temp_dir, file_name)

            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            print(f"成功将DataFrame保存到临时文件: {file_path}")
        return df


        
        
    def nearby(self, city, lat: float, lon: float, topk: int = None, dist: float = 5) -> DataFrame:
        distance = [geodesic((lat, lon), (x, y)).km for x, y in zip(
            self.data[city]['latitude'], self.data[city]['longitude'])]
        tmp = self.data[city].copy()
        tmp['distance'] = distance
        if dist is not None:
            tmp = tmp[tmp['distance'] < dist]
        tmp = tmp.sort_values(by=['distance'])
        if topk is not None:
            return tmp.head(topk)
        return tmp

# class Accommodations_new:

#     def __init__(self, base_path: str = "../../database/hotels/"):
#         curdir = os.path.dirname(os.path.realpath(__file__))
#         city_list = [
#             "beijing", "shanghai", "nanjing",
#             "suzhou", "hangzhou", "shenzhen",
#             "chengdu", "wuhan", "guangzhou",
#             "chongqing"]
#         data_path_list = [os.path.join(
#             curdir, f"{base_path}/{city}/hotel_{city}_uniq.csv") for city in city_list]
#         self.data = {}
#         for i, city in enumerate(city_list):
#             self.data[city] = pd.read_csv(data_path_list[i]).dropna()[
#                 ['hotelName', 'featureHotelType', 'poi_coordinate', 'miniPrice', 'miniPriceRoom', 'hotelImage']]
#             self.data[city]['Latitude'] = self.data[city]['poi_coordinate'].apply(
#                 lambda x: float(x.split("'latitude': ")[
#                                 1].split(",")[0].replace("'", ""))
#             )
#             self.data[city]['Longitude'] = self.data[city]['poi_coordinate'].apply(
#                 lambda x: float(x.split("'longitude': ")[1].split(",")[
#                                 0].replace("'", "").replace("}", ""))
#             )
#             self.data[city]["Price"] = self.data[city]["miniPrice"]
#             self.data[city]['numBed'] = self.data[city]['miniPriceRoom'].apply(
#                 lambda x: 1 if ("大床" in x) or ("单人" in x) else 2
#             )
#             self.data[city] = self.data[city].drop(
#                 ["miniPrice", "miniPriceRoom"], axis=1)
#             self.data[city] = self.data[city].rename(
#                 columns={"hotelName": "Name"})
#             self.data[city] = self.data[city].drop(["poi_coordinate"], axis=1)
#             self.data[city].columns = [x.lower()
#                                        for x in self.data[city].columns]
#             # self.data[city] = self.data[city].drop(
#             #     ["featurehoteltype"], axis=1)

#         self.key_type_tuple_list = {}
#         for city in city_list:
#             self.key_type_tuple_list[city] = []
#             for key in self.data[city].keys():
#                 self.key_type_tuple_list[city].append(
#                     (key, type(self.data[city].iloc[0][key]))
#                 )
#         city_cn_list = ["北京", "上海", "南京", "苏州", "杭州",
#                         "深圳", "成都", "武汉", "广州", "重庆"]
#         def to_float(x):
#             try:
#                 return float(x)
#             except:
#                 return 0.0
#         for i, city in enumerate(city_list):
#             self.data[city_cn_list[i]] = self.data.pop(city)
#             self.key_type_tuple_list[city_cn_list[i]
#                                      ] = self.key_type_tuple_list.pop(city)
#             self.data[city_cn_list[i]]['price'] = self.data[city_cn_list[i]]['price'].apply(
#                 to_float)    
#         # print("Accommodations loaded.")

#     def keys(self, city):
#         return self.key_type_tuple_list[city]

#     def select(self, city, key, func: Callable) -> DataFrame:
#         if key not in self.data[city].keys():
#             return "Key not found."
#         bool_list = [func(x) for x in self.data[city][key]]
#         return self.data[city][bool_list]

#     def nearby(self, city, lat: float, lon: float, topk: int = None, dist: float = 5) -> DataFrame:
#         distance = [geodesic((lat, lon), (x, y)).km for x, y in zip(
#             self.data[city]['latitude'], self.data[city]['longitude'])]
#         tmp = self.data[city].copy()
#         tmp['distance'] = distance
#         if dist is not None:
#             tmp = tmp[tmp['distance'] < dist]
#         tmp = tmp.sort_values(by=['distance'])
#         if topk is not None:
#             return tmp.head(topk)
#         return tmp

if __name__ == "__main__":
    import json
    # AccommodationsAPI = Accommodations_new()
    # city_en_list = ["beijing", "shanghai", "nanjing", "suzhou", "hangzhou", "shenzhen", "chengdu", "wuhan", "guangzhou", "chongqing"]
    # city_cn_list = ["北京", "上海", "南京", "苏州", "杭州", "深圳", "成都", "武汉", "广州", "重庆"]
    # for i, city in enumerate(city_cn_list):
    #     json_data_list=[]
    #     path = f"database/image_url/{city_en_list[i]}/accommodation.json"
    #     for index, row in AccommodationsAPI.data[city].iterrows():
    #         json_data = {
    #             "name": row["name"],
    #             "image_url": row["hotelimage"]
    #         }
    #         json_data_list.append(json_data)
    #     with open(path, "w", encoding="utf-8") as f:
    #         json.dump(json_data_list, f, ensure_ascii=False, indent=4)

    # def query_key(key):
    #     print("query key {}".format(key))
    #     print(AccommodationsAPI.get_info(key))

    # for key in ["Price", "numBed", "hotelName"]:
    #     query_key(key)

    # def query_nearby(lat=32.040158, lon=118.823291):

    #     print("query nearby ({}, {}): ".format(lat, lon))
    #     print(AccommodationsAPI.nearby(lat=lat, lon=lon, topk=None, dist=2))

    # query_nearby()

    # print(AccommodationsAPI.select("numBed", 2))

    # print(AccommodationsAPI.data['featureHotelType'].unique())

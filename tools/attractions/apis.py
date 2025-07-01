import pandas as pd
from pandas import DataFrame
from typing import Callable
import os
from geopy.distance import geodesic
from tools.base_api import search_keywords
import sys
from datetime import datetime

# class Attractions:
#     def __init__(self, path: str = "../../database/attractions/nanjing/attractions_nanjing.csv"):
#         curdir = os.path.dirname(os.path.realpath(__file__))
#         self.path = os.path.join(curdir, path)

#         self.data = pd.read_csv(self.path).dropna()[
#             ['id', 'Name', 'Type', 'Latitude', 'Longitude', 'Phone', 'dateDesc', 'OpenTime', 'EndTime', 'Price', 'RecommendMinTime', 'RecommendMaxTime']]
#         self.data['OpenTime'] = self.data['OpenTime'].apply(
#             lambda x: float(x.split(":")[0]) + float(x.split(":")[1]) / 60)
#         self.data['EndTime'] = self.data['EndTime'].apply(
#             lambda x: float(x.split(":")[0]) + float(x.split(":")[1]) / 60)
#         self.type_list = self.data['Type'].unique()
#         self.data.columns = [x.lower() for x in self.data.columns]
#         self.data = self.data.drop(["phone", "datedesc"], axis=1)
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

#     def id_is_open(self, id: int, time: str) -> bool:
#         open_time = self.data['opentime'][id]
#         end_time = self.data['endtime'][id]
#         time = float(time.split(":")[0]) + float(time.split(":")[1]) / 60
#         if open_time < end_time:
#             return open_time <= time <= end_time
#         else:
#             return open_time <= time or time <= end_time

#     def nearby(self, lat: float, lon: float, topk: int = None, dist=2) -> DataFrame:
#         distance = [geodesic((lat, lon), (x, y)).km for x, y in zip(
#             self.data['latitude'], self.data['longitude'])]
#         tmp = self.data.copy()
#         tmp['distance'] = distance
#         tmp = tmp.sort_values(by=['distance'])
#         if topk is None:
#             return tmp[tmp['distance'] <= dist]
#         return tmp[tmp['distance'] <= dist].head(topk)

#     def get_type_list(self):
#         return self.type_list


class Attractions:
    def __init__(
        self, base_path: str = "../../database/attractions", need_tag: bool = False
    ):
        city_list = [
            "beijing",
            "shanghai",
            "nanjing",
            "suzhou",
            "hangzhou",
            "shenzhen",
            "chengdu",
            "wuhan",
            "guangzhou",
            "chongqing",
        ]
        curdir = os.path.dirname(os.path.realpath(__file__))
        data_path_list = [
            os.path.join(curdir, f"{base_path}/{city}/attractions.csv")
            for city in city_list
        ]
        if need_tag:
            data_path_list = [
                os.path.join(curdir, f"{base_path}/{city}/attractions_tag.csv")
                for city in city_list
            ]
        # name为key value为dataframe
        self.data = {}
        # id,Name,Lat,Lon,Price,CuisineName,WeekdayOpenTime,WeekdayCloseTime,RecommendedFood
        for i, city in enumerate(city_list):
            self.data[city] = pd.read_csv(data_path_list[i])
            self.data[city].columns = [x.lower() for x in self.data[city].columns]
        self.key_type_tuple_list_map = {}
        for city in city_list:
            self.key_type_tuple_list_map[city] = []
            for key in self.data[city].keys():
                self.key_type_tuple_list_map[city].append(
                    (key, type(self.data[city][key][0]))
                )
        self.type_list_map = {}
        for city in city_list:
            self.type_list_map[city] = self.data[city]["type"].unique()
        city_cn_list = [
            "北京",
            "上海",
            "南京",
            "苏州",
            "杭州",
            "深圳",
            "成都",
            "武汉",
            "广州",
            "重庆",
        ]

        def to_float(x):
            try:
                return float(x)
            except:
                return 0.0

        for i, city in enumerate(city_list):
            self.data[city_cn_list[i]] = self.data.pop(city)
            self.key_type_tuple_list_map[city_cn_list[i]] = (
                self.key_type_tuple_list_map.pop(city)
            )
            self.type_list_map[city_cn_list[i]] = self.type_list_map.pop(city)
            self.data[city_cn_list[i]]["price"] = self.data[city_cn_list[i]][
                "price"
            ].apply(to_float)
        # print("Attractions loaded.")

    def keys(self, city: str):
        return self.key_type_tuple_list_map[city]

    def select(self, city, keywords=None, key=None, func=None):
        # 如果没有提供关键词，使用默认值
        if keywords is None:
            keywords = "景点"

        print(f"搜索关键词: {keywords}")

        # 调用API
        result = search_keywords(keywords=keywords, region=city)

        if not result or "pois" not in result or not result["pois"]:
            return pd.DataFrame()  # 返回空DataFrame

        # 转换API返回的结果为DataFrame
        attractions_data = []
        url_list = None
        for poi in result["pois"]:
            if "photos" in poi and poi["photos"] is not None:
                url_list = [
                    photo["url"]
                    for photo in poi["photos"]
                    if photo and isinstance(photo, dict) and "url" in photo
                ]
            location = poi["location"].split(",")
            attraction_data = {
                "name": poi["name"],
                "id": poi["id"],
                "type": poi.get("type", ""),  # 类型
                "address": poi.get("address", ""),
                "rating": poi.get("business", {}).get("rating", ""),
                "price": poi.get("business", {}).get("cost", ""),
                "ood_type": poi.get("business", {}).get("tag", ""),
                "open_time": poi.get("business", {}).get("opentime_week", ""),
                "indoor": poi.get("indoor", {}).get("indoor_map", ""),
                "latitude": float(location[1]) if len(location) > 1 else 0.0,
                "longitude": float(location[0]) if len(location) > 0 else 0.0,
                "phone": poi.get("business", {}).get("tel", ""),
                "photos": url_list,
                "city": city,
            }

            attractions_data.append(attraction_data)

        df = pd.DataFrame(attractions_data)
        if key and func and key in df.columns:
            bool_list = [func(x) for x in df[key]]
            df = df[bool_list]
        if not df.empty:  # 确保DataFrame不为空才保存
            temp_dir = "temp_csv_files"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"attractions_{city}_{timestamp}.csv"
            file_path = os.path.join(temp_dir, file_name)

            df.to_csv(file_path, index=False, encoding="utf-8-sig")

            print(f"成功将DataFrame保存到临时文件: {file_path}")
        return df

    def id_is_open(self, city: str, id: int, time: str) -> bool:
        open_time = self.data[city]["opentime"][id]
        end_time = self.data[city]["endtime"][id]
        open_time = float(open_time.split(":")[0]) + float(open_time.split(":")[1]) / 60
        end_time = float(end_time.split(":")[0]) + float(end_time.split(":")[1]) / 60
        time = float(time.split(":")[0]) + float(time.split(":")[1]) / 60
        if open_time < end_time:
            return open_time <= time <= end_time
        else:
            return open_time <= time or time <= end_time

    def nearby(
        self, city: str, lat: float, lon: float, topk: int = None, dist=2
    ) -> DataFrame:
        distance = [
            geodesic((lat, lon), (x, y)).km
            for x, y in zip(self.data[city]["latitude"], self.data[city]["longitude"])
        ]
        tmp = self.data[city].copy()
        tmp["distance"] = distance
        tmp = tmp.sort_values(by=["distance"])
        if topk is None:
            return tmp[tmp["distance"] <= dist]
        return tmp[tmp["distance"] <= dist].head(topk)

    def get_type_list(self, city: str):
        return self.type_list_map[city]


if __name__ == "__main__":
    a = Attractions()
    print(a.get_type_list("nanjing"))
    # print(a.data)
    # print(a.get_info("Name"))
    # info_list, _ = a.get_info("Name")
    # print(a.get_info_for_index(info_list, 0))
    # print(a.get_info_for_index(info_list, [0, 1]))
    # print(a.nearby(a.data.iloc[0]['Latitude'], a.data.iloc[0]['Longitude']))
    # print(a.select("Name", "夫子庙"))
    # print(a.id_is_open(0, "10:00"))
    # print(a.select('Type', lambda x: x == '公园'))
    # print(a.data)

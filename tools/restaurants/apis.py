import pandas as pd
from pandas import DataFrame
from typing import Callable
import os
from geopy.distance import geodesic


# class Restaurants:
#     def __init__(self, path="../../database/restaurants/nanjing/restaurants_nanjing.csv"):
#         curdir = os.path.dirname(os.path.realpath(__file__))
#         self.path = os.path.join(curdir, path)
#         self.data = pd.read_csv(self.path).dropna(subset=[
#             'id', 'Name', 'Lat', 'Lon', 'Price', 'CuisineName', 'WeekdayOpenTime', 'WeekdayCloseTime', 'WeekendOpenTime', 'WeekendCloseTime'])[['id', 'Name', 'Lat', 'Lon', 'Price', 'Address', 'CuisineName', 'WeekdayOpenTime', 'WeekdayCloseTime', 'WeekendOpenTime', 'WeekendCloseTime', 'RecommendedFood', 'Tel']]
#         self.data['WeekdayOpenTime'] = self.data['WeekdayOpenTime'].apply(
#             lambda x: -1 if x == '不营业' else float(x.split(":")[0]) + float(x.split(":")[1]) / 60)
#         self.data['WeekdayCloseTime'] = self.data['WeekdayCloseTime'].apply(
#             lambda x: -1 if x == '不营业' else float(x.split(":")[0]) + float(x.split(":")[1]) / 60)
#         self.data['WeekendOpenTime'] = self.data['WeekendOpenTime'].apply(
#             lambda x: -1 if x == '不营业' else float(x.split(":")[0]) + float(x.split(":")[1]) / 60)
#         self.data['WeekendCloseTime'] = self.data['WeekendCloseTime'].apply(
#             lambda x: -1 if x == '不营业' else float(x.split(":")[0]) + float(x.split(":")[1]) / 60)
#         self.data['RecommendedFood'] = self.data['RecommendedFood'].fillna("")
#         self.cuisine_list = self.data['CuisineName'].unique()
#         self.data.rename(columns={'CuisineName': 'Cuisine'}, inplace=True)
#         self.data.rename(columns={'Lat': 'Latitude'}, inplace=True)
#         self.data.rename(columns={'Lon': 'Longitude'}, inplace=True)
#         self.data.columns = [x.lower() for x in self.data.columns]
#         self.data = self.data.drop(
#             ["tel", "address", "weekendopentime", "weekendclosetime"], axis=1)
#         self.key_type_tuple_list = []
#         for key in self.data.keys():
#             self.key_type_tuple_list.append((key, type(self.data[key][0])))
#         print("Restaurants loaded.")

#     def keys(self):
#         return self.key_type_tuple_list

#     def select(self, key, func: Callable) -> DataFrame:
#         if key not in self.data.keys():
#             return "Key not found."
#         bool_list = [func(x) for x in self.data[key]]
#         return self.data[bool_list]

#     def id_is_open(self, id: int, time: str, day_num: int) -> bool:
#         # if day_num < 1 or day_num > 7:
#         #     return "invalid day number"
#         # if day_num < 6:
#         #     open_time = self.data['weekdayopentime'][id]
#         #     end_time = self.data['weekdayclosetime'][id]
#         # else:
#         #     open_time = self.data['weekendopentime'][id]
#         #     end_time = self.data['weekendclosetime'][id]
#         open_time = self.data['weekdayopentime'][id]
#         end_time = self.data['weekdayclosetime'][id]
#         if open_time == -1 or end_time == -1:
#             return False
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

#     def restaurants_with_recommended_food(self, food: str):
#         return self.data[self.data['recommendedfood'].str.contains(food)]

#     def get_cuisine_list(self):
#         return self.cuisine_list

class Restaurants:
    def __init__(self, base_path: str = "../../database/restaurants"):
        city_list = [
            "beijing", "shanghai", "nanjing",
            "suzhou", "hangzhou", "shenzhen",
            "chengdu", "wuhan", "guangzhou",
            "chongqing"]
        self.data = {}
        curdir = os.path.dirname(os.path.realpath(__file__))
        for city in city_list:
            path = os.path.join(curdir, base_path, city,
                                "restaurants_" + city + ".csv")
            self.data[city] = pd.read_csv(path)
        for city in city_list:
            self.data[city].columns = [x.lower()
                                       for x in self.data[city].columns]
            self.data[city].rename(
                columns={'cuisinename': 'cuisine'}, inplace=True)
        self.key_type_tuple_list_map = {}
        for city in city_list:
            self.key_type_tuple_list_map[city] = []
            for key in self.data[city].keys():
                self.key_type_tuple_list_map[city].append(
                    (key, type(self.data[city][key][0]))
                )
        self.cuisine_list_map = {}
        for city in city_list:
            self.cuisine_list_map[city] = self.data[city]['cuisine'].unique()
        city_cn_list = ["北京", "上海", "南京", "苏州", "杭州",
                        "深圳", "成都", "武汉", "广州", "重庆"]
        def to_float(x):
            try:
                return float(x)
            except:
                return 0.0
        for i, city in enumerate(city_list):
            self.data[city_cn_list[i]] = self.data.pop(city)
            self.key_type_tuple_list_map[city_cn_list[i]
                                         ] = self.key_type_tuple_list_map.pop(city)
            self.cuisine_list_map[city_cn_list[i]
                                  ] = self.cuisine_list_map.pop(city)
            self.data[city_cn_list[i]]['price'] = self.data[city_cn_list[i]]['price'].apply(
                to_float)
        # print("Restaurants loaded.")

    def keys(self, city: str):
        return self.key_type_tuple_list_map[city]

    def select(self, city: str, key, func: Callable) -> DataFrame:
        if key not in self.data[city].keys():
            return "Key not found."
        bool_list = [func(x) for x in self.data[city][key]]
        return self.data[city][bool_list]

    def id_is_open(self, city: str, id: int, time: str) -> bool:
        open_time = self.data[city]['weekdayopentime'][id]
        open_time = -1 if open_time == '不营业' else float(
            open_time.split(":")[0]) + float(open_time.split(":")[1]) / 60
        end_time = self.data[city]['weekdayclosetime'][id]
        end_time = -1 if end_time == '不营业' else float(
            end_time.split(":")[0]) + float(end_time.split(":")[1]) / 60
        time = float(time.split(":")[0]) + float(time.split(":")[1]) / 60
        if open_time == -1 or end_time == -1:
            return False
        if open_time < end_time:
            return open_time <= time <= end_time
        else:
            return open_time <= time or time <= end_time

    def nearby(self, city: str, lat: float, lon: float, topk: int = None, dist=2) -> DataFrame:
        distance = [geodesic((lat, lon), (x, y)).km for x, y in zip(
            self.data[city]['lat'], self.data[city]['lon'])]
        tmp = self.data[city].copy()
        tmp['distance'] = distance
        tmp = tmp.sort_values(by=['distance'])
        if topk is None:
            return tmp[tmp['distance'] <= dist]
        return tmp[tmp['distance'] <= dist].head(topk)

    def restaurants_with_recommended_food(self, city: str, food: str):
        return self.data[city][self.data[city]['recommendedfood'].str.contains(food)]

    def get_cuisine_list(self, city: str):
        return self.cuisine_list_map[city]


if __name__ == "__main__":
    a = Restaurants()

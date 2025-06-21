from math import radians, sin, cos, sqrt, atan2
import os
import pandas as pd
from pandas import DataFrame
from typing import Callable
city_list = ["上海", "北京", "深圳", "广州", "重庆", "苏州", "成都", "杭州", "武汉", "南京"]
airport_names = {
    "上海": ["上海浦东国际机场", "上海虹桥国际机场"],
    "北京": ["北京首都国际机场", "北京大兴国际机场"],
    "深圳": ["深圳宝安国际机场"],
    "广州": ["广州白云国际机场"],
    "重庆": ["重庆江北国际机场"],
    "苏州": [],
    "成都": ["成都双流国际机场", "成都天府国际机场"],
    "杭州": ["杭州萧山国际机场"],
    "武汉": ["武汉天河国际机场"],
    "南京": ["南京禄口国际机场"]
}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2)**2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def parse_duration(duration):
    hours, minutes = map(int, duration.replace(
        "小时", " ").replace("分钟", "").split(" "))
    return round(hours + minutes / 60, 2)


def get_price_dict():
    city_coords = {
        "上海": (31.2304, 121.4737),
        "北京": (39.9042, 116.4074),
        "深圳": (22.5431, 114.0579),
        "广州": (23.1291, 113.2644),
        "重庆": (29.5630, 106.5516),
        "苏州": (31.2983, 120.5832),
        "成都": (30.5728, 104.0668),
        "杭州": (30.2741, 120.1551),
        "武汉": (30.5928, 114.3055),
        "南京": (32.0603, 118.7969)
    }

    price_per_km = {
        "T": 0.5,
        "K": 0.4,
        "G": 0.65,
        "D": 0.60,
        "Z": 0.5,
        "C": 0.45
    }

    price_table = {}
    for city1 in city_coords:
        for city2 in city_coords:
            if city1 != city2:
                lat1, lon1 = city_coords[city1]
                lat2, lon2 = city_coords[city2]
                distance = haversine(lat1, lon1, lat2, lon2)
                price_table[(city1, city2)] = {
                    model: round(distance * rate, 2)
                    for model, rate in price_per_km.items()
                }

    return price_table


class IntercityTransport:
    def __init__(self, path: str = "../../database/intercity_transport/"):
        curdir = os.path.dirname(os.path.realpath(__file__))
        self.base_path = os.path.join(curdir, path)
        self.train_price_table = get_price_dict()
        self.airplane_path = self.base_path+'airplane.jsonl'
        self.airplane_df = pd.read_json(
            self.airplane_path, lines=True, keep_default_dates=False)

        self.train_df_dict = {}

        for start_city in city_list:
            for end_city in city_list:
                if start_city == end_city:
                    continue
                train_path = self.base_path+'train/' + \
                    'from_{}_to_{}.json'.format(start_city, end_city)
                train_df = pd.read_json(train_path)
                if train_df.empty == True:
                    self.train_df_dict[(start_city, end_city)] = train_df
                    continue
                train_df.drop(columns=["始发站", "终点站"], inplace=True)
                train_df.rename(columns={
                    "车次": "TrainID",
                    "列车类型": "TrainType",
                    "出发站": "From",
                    "到达站": "To",
                    "发车时间": "BeginTime",
                    "到达时间": "EndTime",
                    "耗时": "Duration"
                }, inplace=True)

                train_df['From'] = train_df['From']+"站"
                train_df['To'] = train_df['To']+"站"
                train_df['BeginTime'] = train_df['BeginTime'].str.split(
                    '\n').str[0]
                train_df['EndTime'] = train_df['EndTime'].str.split(
                    '\n').str[0]

                train_df['Duration'] = train_df['Duration'].apply(
                    parse_duration)

                cost_dict = self.train_price_table[(start_city, end_city)]
                train_df['Cost'] = train_df['TrainID'].apply(
                    lambda x: cost_dict.get(x[0], None))
                train_df = train_df[["TrainID", "TrainType", "From",
                                     "To", "BeginTime", "EndTime", "Duration", "Cost"]]
                self.train_df_dict[(start_city, end_city)] = train_df

    def select(self, start_city, end_city, intercity_type) -> DataFrame:
        # intercity_type=='train' | 'airplane'
        if intercity_type == 'airplane':

            if len(self.airplane_df) == 0:
                return None

            filtered_flights = self.airplane_df[(self.airplane_df['From'].str.contains(start_city)) & (
                self.airplane_df['To'].str.contains(end_city))]
            sorted_flights = filtered_flights.sort_values(
                by='BeginTime').reset_index(drop=True)
            return sorted_flights
        if intercity_type == 'train':

            if len(self.train_df_dict[(start_city, end_city)]) == 0:
                return None

            filtered_trains = self.train_df_dict[(start_city, end_city)]
            sorted_trains = filtered_trains.sort_values(
                by='BeginTime').reset_index(drop=True)
            return sorted_trains


if __name__ == "__main__":
    a = IntercityTransport()
    city_en_list = ["shanghai", "beijing", "shenzhen", "guangzhou",
                    "chongqing", "suzhou", "chengdu", "hangzhou", "wuhan", "nanjing"]
    str_list = []
    for i in range(len(city_list)):
        for j in range(i+1, len(city_list)):
            # "('{}','{}','{}',{})".format(city_list[i],city_list[j],'airplane',len(a.select(city_list[i],city_list[j],'airplane')))
            # str1 = "('{}','{}','{}',{})".format(city_en_list[i],city_en_list[j],'train',len(a.select(city_list[i],city_list[j],'train')))
            # str2 = "('{}','{}','{}',{})".format(city_en_list[i],city_en_list[j],'airplane',len(a.select(city_list[i],city_list[j],'airplane')))
            # str3 = "('{}','{}','{}',{})".format(city_en_list[j],city_en_list[i],'train',len(a.select(city_list[j],city_list[i],'train')))
            # str4 = "('{}','{}','{}',{})".format(city_en_list[j],city_en_list[i],'airplane',len(a.select(city_list[j],city_list[i],'airplane')))
            # print(str1+",")
            # print(str3+",")
            # print(str2+",")
            # print(str4+",")
            tmp_len = 0

            tmp = a.select(city_list[i], city_list[j], 'train')
            if not isinstance(tmp, DataFrame):
                tmp_len = 0
            else:
                tmp_len = len(tmp)
            if tmp_len > 0:
                str_list.append("('{}','{}','{}',{})".format(
                    city_en_list[i], city_en_list[j], 'train', tmp_len))

            tmp = a.select(city_list[j], city_list[i], 'train')
            if not isinstance(tmp, DataFrame):
                tmp_len = 0
            else:
                tmp_len = len(tmp)
            if tmp_len > 0:
                str_list.append("('{}','{}','{}',{})".format(
                    city_en_list[j], city_en_list[i], 'train', tmp_len))

            tmp = a.select(city_list[i], city_list[j], 'flight')
            if not isinstance(tmp, DataFrame):
                tmp_len = 0
            else:
                tmp_len = len(tmp)
            if tmp_len > 0:
                str_list.append("('{}','{}','{}',{})".format(
                    city_en_list[i], city_en_list[j], 'airplane', tmp_len))

            tmp = a.select(city_list[j], city_list[i], 'flight')
            if not isinstance(tmp, DataFrame):
                tmp_len = 0
            else:
                tmp_len = len(tmp)
            if tmp_len > 0:
                str_list.append("('{}','{}','{}',{})".format(
                    city_en_list[j], city_en_list[i], 'airplane', tmp_len))

    # ,\n   ".join(str_list)
    print(",\n".join(str_list))

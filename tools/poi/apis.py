import os
import json
from fuzzywuzzy import process, fuzz


class Poi:
    def __init__(self, base_path: str = "../../database/poi/"):

        city_list = [
            "beijing", "shanghai", "nanjing",
            "suzhou", "hangzhou", "shenzhen",
            "chengdu", "wuhan", "guangzhou",
            "chongqing"]
        curdir = os.path.dirname(os.path.realpath(__file__))
        data_path_list = [os.path.join(
            curdir, f"{base_path}/{city}/poi.json") for city in city_list]
        self.data = {}
        # self.data = json.load(open(self.path, "r", encoding="utf-8"))
        # self.data = [(x["name"], tuple(x["position"])) for x in self.data]
        for i, city in enumerate(city_list):
            # print(f"Loading {city}...")
            self.data[city] = json.load(
                open(data_path_list[i], "r", encoding="utf-8"))
            self.data[city] = [(x["name"], tuple(x["position"]))
                               for x in self.data[city]]
        city_cn_list = ["北京", "上海", "南京", "苏州", "杭州",
                        "深圳", "成都", "武汉", "广州", "重庆"]
        for i, city in enumerate(city_list):
            self.data[city_cn_list[i]] = self.data.pop(city)
        # print("Poi loaded.")

    def search(self, city: str, name: str):
        # 搜索最匹配的poi的坐标
        # name, _ = process.extractOne(
        #     name, [x[0] for x in self.data[city]])
        # # print(f"匹配到的poi name: {name}")
        # for x in self.data[city]:
        #     if x[0] == name:
        #         return x[1]
        # 严格匹配
        for x in self.data[city]:
            if x[0] == name:
                return x[1]
        return None


def test():
    poi = Poi()
    while True:
        query = input("请输入查询的poi名称：")
        if query == "exit":
            return
        print(poi.search('南京', query))


if __name__ == "__main__":
    test()

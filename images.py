import json
import os

def name_url_to_kv(nu_dict_list):
    kv_dict = {}
    for nu_dict in nu_dict_list:
        kv_dict[nu_dict["name"]] = nu_dict["image_url"].split(".jpg")[0]+".jpg"
    return kv_dict

class image_url():
    def __init__(self):
        city_list = [
        "beijing", "shanghai", "nanjing",
        "suzhou", "hangzhou", "shenzhen",
        "chengdu", "wuhan", "guangzhou",
        "chongqing"]
        self.data = {}
        poi_types=["restaurant", "attraction","accommodation"]
        for poi_type in poi_types:
            self.data[poi_type] = {}
            for city in city_list:
                self.data[poi_type][city] = {}
                with open(f"database/image_url/{city}/{poi_type}.json", "r", encoding="utf-8") as f:
                    self.data[poi_type][city] = name_url_to_kv(json.load(f))
                    
    def get_image_url(self, city, poi_type, name):
        if poi_type in ["breakfast", "lunch", "dinner"]:
            poi_type = "restaurant"
            try:
                res = self.data[poi_type][city][name]
            except:
                poi_type = "accommodation"
                res = self.data[poi_type][city][name]
            return res
        if poi_type not in self.data:
            return ""
        return self.data[poi_type][city][name]

__tool__ = image_url()
__cn__list__ = ["北京","上海","南京","苏州","杭州","深圳","成都","武汉","广州","重庆"]
__en__list__ = ["beijing","shanghai","nanjing","suzhou","hangzhou","shenzhen","chengdu","wuhan","guangzhou","chongqing"]
def get_image_url(city, poi_type, name):
    print(city,poi_type,name)
    if city in __cn__list__:
        city = __en__list__[__cn__list__.index(city)]
    return __tool__.get_image_url(city, poi_type, name)

if __name__ == "__main__":
    print(__tool__.data["accommodation"]["hangzhou"].keys())
    print(get_image_url("杭州", "accommodation", "杭州西湖湖滨平海路亚朵酒店"))
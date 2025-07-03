
import random
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta
import json
# no airplane for 苏州
city_list=["上海","北京","深圳","广州","重庆","成都","杭州","武汉","南京"]
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

    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

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

def generate_flight_info(city_list, city_coords):
    flight_info = []
    flight_id = 1
    
    for i in range(len(city_list)):
        for j in range(len(city_list)):
            if j==i:
                continue
            city1 = city_list[i]
            city2 = city_list[j]
            coord1 = city_coords[city1]
            coord2 = city_coords[city2]
            
            distance = haversine(coord1[0], coord1[1], coord2[0], coord2[1])
            
            for _ in range(10):  # Generate 5 flights for each city pair
                # Generate random departure time
                hour = random.randint(0, 23)
                minute = random.randint(0, 59)
                departure_time = f"{hour:02}:{minute:02}"
                
                # Estimate flight duration (assuming average speed of 800 km/h)
                duration_hours = round(distance / 800,2)
                
                if distance<500:
                    print(city1,city2)
                    price = round(distance * 1.0, 2)+random.uniform(-50, 50)
                else:
                    price = round(distance * 0.5, 2)+random.uniform(-50, 50)
                departure_datetime = datetime.strptime(departure_time, "%H:%M")
                duration_timedelta = timedelta(hours=duration_hours)
                arrival_datetime=departure_datetime+duration_timedelta
                flight_info.append({
                    "FlightID": f"FL{flight_id:03}",
                    "From": random.choice(airport_names[city1]),
                    "To": random.choice(airport_names[city2]),
                    "BeginTime": departure_time,
                    "EndTime":arrival_datetime.strftime("%H:%M"),
                    "Duration": duration_hours,
                    "Cost": round(price,2)
                })
                
                flight_id += 1
    
    return flight_info

flights = generate_flight_info(city_list, city_coords)
jsonl_file_path='../../database/intercity_transport/airplane.jsonl'
with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
    for entry in flights:
        jsonl_file.write(json.dumps(entry,ensure_ascii=False) + '\n')


import os
import json
import heapq
from fuzzywuzzy import process
from geopy.distance import geodesic
from datetime import datetime, timedelta
from tools.base_api import search_routine
import sys
sys.path.append('../../')
from tools.poi.apis import Poi

verbose = True

curdir = os.path.dirname(os.path.realpath(__file__))
SUBWAY_PATH = os.path.join(
    curdir, "../../database/transportation/subways.json")
city_list = ["shanghai", "beijing", "shenzhen", "guangzhou",
             "chongqing", "suzhou", "chengdu", "hangzhou", "wuhan", "nanjing"]
city_list_chinese = ["上海", "北京", "深圳", "广州",
                     "重庆", "苏州", "成都", "杭州", "武汉", "南京"]


def get_lines_and_stations(city):
    stations_all = []
    metro_lines = {}
    with open(SUBWAY_PATH, 'r', encoding='utf-8') as file:
        subway_data = json.load(file)
    for line in subway_data[city]:
        metro_lines[line['name']] = []
        for station in line['stations']:
            lat, lon = map(float, station['position'].split(','))
            metro_lines[line['name']].append(station['name'])
            stations_all.append({
                'name': station['name'],
                'position': (lon, lat)
            })
    station_to_line = {}
    for line, stations in metro_lines.items():
        for station in stations:
            station_to_line[station] = line
    return stations_all, metro_lines, station_to_line


city_stations_dict = {}
city_lines_dict = {}
city_station_to_line = {}
for city in city_list:
    stations_all, metro_lines, station_to_line = get_lines_and_stations(city)
    city_stations_dict[city] = stations_all
    city_lines_dict[city] = metro_lines
    city_station_to_line[city] = station_to_line


def build_graph(metro_lines):
    graph = {}
    for line, stations in metro_lines.items():
        for i in range(len(stations)):
            if stations[i] not in graph:
                graph[stations[i]] = []
            if i > 0:
                graph[stations[i]].append(stations[i-1])
            if i < len(stations) - 1:
                graph[stations[i]].append(stations[i+1])
    return graph


graphs = {}
for city in city_list:
    graphs[city] = build_graph(city_lines_dict[city])

poi_search = Poi()


def GoTo(city, locationA, locationB, start_time, transport_type, verbose=True):
    
    print("GoTo: From {} to {}".format(locationA, locationB))
    coordinate_A,citycode_A = poi_search.search_loc(city, locationA)
    coordinate_B,citycode_B = poi_search.search_loc(city, locationB)
    city_cn = city
    # "walk", "metro", "taxi"    
    
    #assert (isinstance(locationA, tuple) or not isinstance(locationB, tuple))
    transports = []
    start_time_in=start_time.replace(':', '-')
    routine=search_routine(coordinate_A, coordinate_B,citycode_A,citycode_B,"0",start_time_in)
    if transport_type == 'walk':
        distance = float(routine['distance']) / 1000.0 #单位是km
        walking_speed = 5.
        time = distance / walking_speed
        cost = 0.
        end_time = add_time(start_time, time)

        transport = {
            "start": locationA_name,
            "end": locationB_name,
            "mode": "walk",
            "start_time": start_time,
            "end_time": end_time,
            "cost": cost,
            "distance": distance
        }
        transports.append(transport)
        if verbose:
            print('Walk Distance {:.3} kilometers, Time {:.3} hour, Cost {}¥'.format(
                distance, time, int(cost)))
        return transports

    elif transport_type == 'taxi':
        distance = float(routine['distance']) / 1000.0 #单位是km
        taxi_speed = 40.  # km/h
        time = distance / taxi_speed  # hours
        cost = calculate_cost_taxi(distance)
        end_time = add_time(start_time, time)
        
        transport = {
            "start": locationA,
            "end": locationB,
            "mode": "taxi",
            # "line": "",
            "start_time": start_time,
            "end_time": end_time,
            "cost": round(cost, 2),
            "distance": round(distance, 2), 
        }
        transports.append(transport)

        if verbose:
            print('Taxi Distance {:.3} kilometers, Time {:.2} hour, Cost {}¥'.format(
                distance, time, int(cost)))
        return transports

    elif transport_type == 'metro':
        if routine['time_cost'].get('end_time', '') == '':#说明距离太近，直接走路过去
            return GoTo(city, locationA, locationB, start_time, "walk", verbose)
        distance = float(routine['distance']) / 1000.0
        transports.append({ 
                "start": locationA,
                "end": locationB,
                "mode": "metro",
                "start_time": start_time,
                "end_time": routine['time_cost'].get('end_time', ''),
                "cost": routine['fee_cost'].get('transit_fee', 0),
                "distance":distance,
                "nL_desc": routine['transit_desc']
            })
        print("line142",transports)
        return transports


def get_line_change(station_to_line, path):
    line_changes = []

    for station in path:
        if station in station_to_line:
            line_changes.append(station_to_line[station])
    # print(path,line_changes)


# def add_time(start_time, hours):
#     initial_time = datetime.strptime(start_time, "%H:%M")
#     end_time = initial_time + timedelta(hours=hours)
#     return end_time.strftime("%H:%M")

def add_time(time1, hours):

    hour, minu = int(time1.split(":")[0]), int(time1.split(":")[1])

    time_delta = int(hours * 60)
    min_new = minu + time_delta

    if min_new >= 60:
        hour_new = hour + int(min_new / 60)
        min_new = min_new % 60
    else:
        hour_new = hour

    if hour_new < 10:
        time_new = "0" + str(hour_new) + ":"
    else:
        time_new = str(hour_new) + ":"
    if min_new < 10:

        time_new = time_new + "0" + str(min_new)
    else:
        time_new = time_new + str(min_new)

    return time_new

def dijkstra(graph, start, end):
    queue = [(0, start, [])]
    seen = set()
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        if node in seen:
            continue
        path = path + [node]
        seen.add(node)
        if node == end:
            return path
        for next_node in graph.get(node, []):
            if next_node not in seen:
                heapq.heappush(queue, (cost + 1, next_node, path))
    return []


def find_shortest_path(graph, start, end):
    return dijkstra(graph, start, end)


def find_nearest_station(location, stations):
    nearest_station = None
    min_distance = float('inf')
    for station in stations:
        distance = geodesic(location, station['position']).kilometers
        if distance < min_distance:
            min_distance = distance
            nearest_station = station
    return nearest_station, min_distance


def calculate_cost_taxi(distance):
    # The starting price is 11 yuan, 2.5 yuan per kilometer within 1.8 kilometers, 3.5 yuan per kilometer within 1.8 kilometers to 10 kilometers, and 4.5 yuan per kilometer above 10 kilometers.
    if distance <= 1.8:
        return 11.
    elif distance <= 10:
        return 11. + (distance - 1.8) * 3.5
    else:
        return 11. + (10 - 1.8) * 3.5 + (distance - 10) * 4.5


def calculate_cost(distance):
    if distance <= 4:
        return 2
    elif distance <= 9:
        return 3
    elif distance <= 14:
        return 4
    elif distance <= 21:
        return 5
    elif distance <= 28:
        return 6
    elif distance <= 37:
        return 7
    elif distance <= 48:
        return 8
    elif distance <= 61:
        return 9
    else:
        extra_distance = distance - 61
        extra_cost = (extra_distance + 14) // 15
        return 9 + extra_cost

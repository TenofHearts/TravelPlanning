import os
import json
import heapq
from fuzzywuzzy import process
from geopy.distance import geodesic
from datetime import datetime, timedelta

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
    if(verbose):
        print("GoTo: From {} to {}".format(locationA, locationB))
    coordinate_A = poi_search.search(city, locationA)
    coordinate_B = poi_search.search(city, locationB)

    city_cn = city

    if city in city_list_chinese:
        city = city_list[city_list_chinese.index(city)]
    # "walk", "metro", "taxi"

    locationA_name, locationB_name = locationA, locationB
    locationA, locationB = coordinate_A, coordinate_B
    
    
    assert (isinstance(locationA, tuple) or not isinstance(locationB, tuple))
    transports = []
    if transport_type == 'walk':
        distance = geodesic(locationA, locationB).kilometers
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
        distance = geodesic(locationA, locationB).kilometers
        taxi_speed = 40.  # km/h
        time = distance / taxi_speed  # hours
        cost = calculate_cost_taxi(distance)
        end_time = add_time(start_time, time)

        transport = {
            "start": locationA_name,
            "end": locationB_name,
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

        graph = graphs[city]

        stationA, distanceA = find_nearest_station(
            locationA, city_stations_dict[city])
        stationB, distanceB = find_nearest_station(
            locationB, city_stations_dict[city])
        if stationA == stationB:
            if verbose:
                print('Too near. Walk.')
            return GoTo(city_cn, locationA_name, locationB_name, start_time, transport_type='walk',verbose=verbose)

        shortest_path = find_shortest_path(
            graph, stationA['name'], stationB['name'])

        if stationA and stationB:
            distance_between_stations = geodesic(
                stationA['position'], stationB['position']).kilometers
            subway_speed = 30.
            time_between_stations = distance_between_stations / subway_speed

            walking_speed = 5.
            timeA = distanceA / walking_speed
            timeB = distanceB / walking_speed
            total_time = timeA + time_between_stations + timeB

            cost = calculate_cost(distance_between_stations)

            distance = distanceA + distance_between_stations + distanceB

            end_timeA = add_time(start_time, timeA)
            end_timeB = add_time(end_timeA, time_between_stations)
            end_time_final = add_time(end_timeB, timeB)

            transports.append({
                "start": locationA_name,
                "end": stationA['name'] + '-地铁站',
                "mode": "walk",
                "start_time": start_time,
                "end_time": end_timeA,
                "cost": 0,
                "distance": round(distanceA, 2)
            })

            transports.append({
                "start": stationA['name'] + '-地铁站',
                "end": stationB['name'] + '-地铁站',
                "mode": "metro",
                "start_time": end_timeA,
                "end_time": end_timeB,
                "cost": cost,
                "distance": round(distance_between_stations, 2)
            })

            transports.append({
                "start": stationB['name'] + '-地铁站',
                "end": locationB_name,
                "mode": "walk",
                "start_time": end_timeB,
                "end_time": end_time_final,
                "cost": 0,
                "distance": round(distanceB, 2)
            })
            if verbose:
                print("Walk: From starting point to metro {}, Distance: {}.".format(
                    stationA['name'] + '-地铁站', distanceA))
                print(
                    f"Subway: From {stationA['name'] + '-地铁站'} to {stationB['name'] + '-地铁站'}: {' -> '.join(shortest_path)}")
                print('The cost of subway: {}¥'.format(cost))
                print("Walk: From metro {} to ending point,  Distance: {}.".format(
                    stationB['name'] + '-地铁站', distanceB))
            return transports
        else:
            raise NotImplementedError


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

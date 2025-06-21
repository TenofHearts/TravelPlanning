
# dowload
import requests
import json
urls={
    'beijing': 'https://map.amap.com/service/subway?_1725194508286&srhdata=1100_drw_beijing.json',
    'guangzhou': 'https://map.amap.com/service/subway?_1725194475224&srhdata=4401_drw_guangzhou.json',
    'shenzhen':'https://map.amap.com/service/subway?_1725194349593&srhdata=4403_drw_shenzhen.json',
    'shanghai':'https://map.amap.com/service/subway?_1725194170991&srhdata=3100_drw_shanghai.json',
    'wuhan':'https://map.amap.com/service/subway?_1725194418932&srhdata=4201_drw_wuhan.json',
    'chongqing':'https://map.amap.com/service/subway?_1725194442027&srhdata=5000_drw_chongqing.json',
    'hangzhou':'https://map.amap.com/service/subway?_1725194554642&srhdata=3301_drw_hangzhou.json', 
    'suzhou': 'https://map.amap.com/service/subway?_1725194587859&srhdata=3205_drw_suzhou.json',
    'chengdu': 'https://map.amap.com/service/subway?_1725194624562&srhdata=5101_drw_chengdu.json',
    'nanjing': 'https://map.amap.com/service/subway?_1725194928582&srhdata=3201_drw_nanjing.json'
}
def download():

    for city, url in urls.items():
        response = requests.get(url)
        response.raise_for_status()  
        file_name = f"../../database/transportation/raw/{city}.json"
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

   
def process(raw_path="../../database/transportation/raw/"):
    lines={}
    for city in urls.keys():
        with open(raw_path+'{}.json'.format(city), 'r', encoding='utf-8') as file:
            data = json.load(file)
        lines[city]=[]
        for line in data['l']:
            new_line={
                'name': line['ln'],
                'stations': []
            }
            for st in line['st']:
                new_station={
                    'name':st['n'],
                    'position': st['sl'],
                }
                new_line['stations'].append(new_station)

            lines[city].append(new_line)

    with open('../../database/transportation/subways.json', 'w', encoding='utf-8') as json_file:
        json.dump(lines, json_file, ensure_ascii=False)

process()
# download()
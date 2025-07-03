import requests
from bs4 import BeautifulSoup
import json

city_list=["上海","北京","深圳","广州","重庆","苏州","成都","杭州","武汉","南京"]


def save_data(city_start,city_end):
    url = "https://www.chalieche.com/search/range/?from={}&to={}".format(city_start,city_end)
    response = requests.get(url)
    response.encoding = 'utf-8' 
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')

    train_schedule = []
    table = soup.find('table')
    if table:
        rows = table.find_all('tr')[1:]  
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 0:
                train_info = {
                    "车次": cols[0].text.strip(),
                    "列车类型": cols[1].text.strip(),
                    "始发站": cols[2].text.strip(),
                    "终点站": cols[3].text.strip(),
                    "出发站": cols[4].text.strip(),
                    "发车时间": cols[5].text.strip(),
                    "到达站": cols[6].text.strip(),
                    "到达时间": cols[7].text.strip(),
                    "耗时": cols[8].text.strip()
                }
                train_schedule.append(train_info)

    with open('../../database/intercity_transport/train/from_{}_to_{}.json'.format(city_start,city_end), 'w', encoding='utf-8') as json_file:
        json.dump(train_schedule, json_file, ensure_ascii=False, indent=4)

    
from tqdm import tqdm

for start_city in tqdm(city_list):
    for end_city in city_list:
        if end_city!=start_city:
            print(start_city,end_city)
            save_data(start_city,end_city)



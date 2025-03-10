import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import urllib.request
from bs4 import BeautifulSoup
import time

obs_stations = pd.read_excel("data/obs_stations.xlsx")
obs_stations = obs_stations.query('ed_y == 9999')
obs_stations = obs_stations.query('気温 == "Y"')
obs_stations = obs_stations.query('地点 == "名古屋"')
print(obs_stations.columns)

start_date = datetime.date(2008, 1, 1)
end_date   = datetime.date(2018, 12, 31)

# Temp_data = pd.DataFrame(index=[], columns=['降水量','降雪量','積雪量','平均気温','最高気温','最低気温','平均風速'])
Temp_data = pd.DataFrame(index=[], columns=['全天日射量'])


date = start_date
while date < end_date:
    Temp_data.loc[date] = -100.0
    date += relativedelta(days=1)

def str2float(weather_data):
    try:
        return float(weather_data)
    except:
        return -100

for i in obs_stations.index:
    Temp_data_s = Temp_data.copy()

    url_y = "https://www.data.jma.go.jp/obd/stats/etrn/view/annually_%s.php?" \
            "prec_no=%d&block_no=%04d&year=&month=&day=&view=" \
                %(str.lower(obs_stations['区分'].loc[i]), obs_stations['府県番号'].loc[i], obs_stations['地点コード'].loc[i])

    html = urllib.request.urlopen(url_y).read()
    soup = BeautifulSoup(html)
    trs = soup.find("table", { "class" : "data2_s" })

    if trs is None:
        continue
    
    tr = trs.findAll('tr')[3]
    tds = tr.findAll('td')[0].findAll('div')[0].findAll('a')

    date = start_date
    # date = datetime.date(max(int(tds[0].string),1872), 1, 1)
    while date < end_date:
        url_m = "https://www.data.jma.go.jp/obd/stats/etrn/view/daily_%s1.php?" \
                "prec_no=%s&block_no=%04d&year=%d&month=%d&day=&view=" \
                    %(str.lower(obs_stations['区分'].loc[i]), obs_stations['府県番号'].loc[i], obs_stations['地点コード'].loc[i], date.year, date.month)

        html = urllib.request.urlopen(url_m).read()
        soup = BeautifulSoup(html)
        trs = soup.find("table", { "class" : "data2_s" })

        if trs is None:
            print(f"\r{obs_stations['地点'].loc[i]}", end="")
            date += relativedelta(months=1)
            continue

        # table の中身を取得
        date_day = date
        print(f"\r{i,obs_stations['地点'].loc[i], obs_stations['地点コード'].loc[i], date_day.year, date_day.month}", end="")
        
        if obs_stations['区分'].loc[i] == 'S':
            for tr in trs.findAll('tr')[4:]:
                tds = tr.findAll('td')
                
                if tds[6].string == None or tds[7].string == None or tds[8].string == None:
                    break
                Temp_data_s['全天日射量'].loc[date_day] = str2float((tds[3].string).split(' ')[0])
                # Temp_data_s['平均気温'].loc[date_day] = str2float((tds[6].string).split(' ')[0])
                # Temp_data_s['最高気温'].loc[date_day] = str2float((tds[7].string).split(' ')[0])
                # Temp_data_s['最低気温'].loc[date_day] = str2float((tds[8].string).split(' ')[0])
                # Temp_data_s['降雪量'].loc[date_day] = str2float((tds[17].string).split(' ')[0])
                # Temp_data_s['積雪量'].loc[date_day] = str2float((tds[18].string).split(' ')[0])
                # Temp_data_s['平均風速'].loc[date_day] = str2float((tds[11].string).split(' ')[0])
                date_day += relativedelta(days=1)
        else:
            for tr in trs.findAll('tr')[3:]:
                tds = tr.findAll('td')
 
                if tds[4].string == None or tds[5].string == None or tds[6].string == None:
                    break
                Temp_data_s['全天日射量'].loc[date_day] = str2float((tds[1].string).split(' ')[0])
                # Temp_data_s['平均気温'].loc[date_day] = str2float((tds[4].string).split(' ')[0])
                # Temp_data_s['最高気温'].loc[date_day] = str2float((tds[5].string).split(' ')[0])
                # Temp_data_s['最低気温'].loc[date_day] = str2float((tds[6].string).split(' ')[0])
                # Temp_data_s['降雪量'].loc[date_day] = str2float((tds[16].string).split(' ')[0])
                # Temp_data_s['積雪量'].loc[date_day] = str2float((tds[17].string).split(' ')[0])
                # Temp_data_s['平均風速'].loc[date_day] = str2float((tds[9].string).split(' ')[0])
                date_day += relativedelta(days=1)

        date += relativedelta(months=1)

        time.sleep(3)

    Temp_data_s.to_csv("data/%d_%s.csv"%(obs_stations['地点コード'].loc[i],obs_stations['地点'].loc[i]))


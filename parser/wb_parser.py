# get_catalogs_wb Получение списка всех каталогов wildberries
# search_category_in_catalog Поиск пользовательской категории в списке каталогов
# get_data_from_json Извлечение данных из json файла с информацией о товарах
# get_content Постраничный сбор данных
# save_excel Сохранение данных в эксель файл при помощи библиотеки pandas
# parser Основная функция выполняющая все выше перечисленные

import requests
import random
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver


def get_description(url):
    driver = webdriver.Chrome('chromedriver.exe')
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    desc = soup.find('p', attrs={'class': 'collapsable__text'}).get_text()
    # driver.close()
    return desc


def get_catalogs_wb():
    url = 'https://www.wildberries.ru/webapi/menu/main-menu-ru-ru.json'
    headers = {'Accept': "*/*", 'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    data = response.json()
    data_list = []
    for d in data:
        try:
            for child in d['childs']:
                try:
                    # if child['name'] in [x['category_name'] for x in data_list]:
                    #     continue
                    category_name = child['name']
                    # получается, что упускаем "женские" и берем только "блузки и рубашки", чтобы не упускать можно
                    # брать "seo", но тк эти категории нам нужны для последующего парсинга, то хз, посмотрим
                    category_url = child['seo']
                    shard = child['shard']  # название каталога
                    query = child['query']  # подразделы
                    data_list.append({
                        'category_name': category_name,
                        'category_url': category_url,
                        'shard': shard,
                        'query': query})
                except:
                    continue
                # под вопросом: нужны ли подгатегории (на начальном этапе)?
                try:
                    for sub_child in child['childs']:
                        category_name = sub_child['name']
                        category_url = sub_child['url']
                        shard = sub_child['shard']
                        query = sub_child['query']
                        data_list.append({
                            'category_name': category_name,
                            'category_url': category_url,
                            'shard': shard,
                            'query': query})
                except:
                    continue
        except:
            continue
    return data_list


def search_category_in_catalog(url: str, catalog_list: list) -> [str, str, str]:
    """
    :param url:
    :param catalog_list:
    :return:
    """
    try:
        for catalog in catalog_list:
            if catalog['category_url'] == url.split('https://www.wildberries.ru')[-1]:
                print(f'найдено совпадение: {catalog["category_name"]}')
                name_category = catalog['category_name']
                shard = catalog['shard']
                query = catalog['query']
                return name_category, shard, query
            else:
                # print('нет совпадения')
                pass
    except:
        print('Данный раздел не найден!')


def get_data_from_json(json_file):
    data_list = []
    for data in json_file['data']['products']:
        data_list.append({
            'item_name': data['name'],
            'id': data['id'],
            'url': f'https://www.wildberries.ru/catalog/{data["id"]}/detail.aspx?targetUrl=BP'
        })
    return data_list


def get_content(shard, query, low_price=None, top_price=None):
    headers = {'Accept': "*/*", 'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    data_list = []
    for page in range(1, 10):
        print(f'Сбор позиций со страницы {page} из 100')
        # url = f'https://catalog.wb.ru/catalog/{shard}/catalog?appType=1&curr=rub&dest=-1075831,-77677,-398551,12358499' \
        #       f'&locale=ru&page={page}&priceU={low_price * 100};{top_price * 100}' \
        #       f'®=0®ions=64,83,4,38,80,33,70,82,86,30,69,1,48,22,66,31,40&sort=popular&spp=0&{query}'
        url = f'https://catalog.wb.ru/catalog/{shard}/catalog?appType=1&curr=rub&dest=-1075831,-77677,-398551,12358499' \
              f'&locale=ru&page={page}&priceU={low_price * 10};{top_price * 10}' \
              f'&reg=0&regions=64,83,4,38,80,33,70,82,86,30,69,1,48,22,66,31,40&sort=popular&spp=0&{query}'
        url = f'https://catalog.wb.ru/catalog/{shard}/catalog?appType=1&curr=rub&dest=-1075831,-77677,-398551,12358499&locale=ru&page={page}&priceU={low_price * 10};{top_price * 10}&reg=0&regions=64,83,4,38,80,33,70,82,86,30,69,1,48,22,66,31,40&sort=popular&spp=0&{query}'
        # url = "https://www.wildberries.ru/catalog/zhenshchinam/odezhda/bluzki-i-rubashki"
        r = requests.get(url, headers=headers)

        data = r.json()
        print(f'Добавлено позиций: {len(get_data_from_json(data))}')
        if len(get_data_from_json(data)) > 0:
            data_list.extend(get_data_from_json(data))
        else:
            print(f'Сбор данных завершен.')
            break
    return data_list


def save_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(f'{filename}.csv', encoding='utf-8-sig')
    print(f'Все сохранено в data/{filename}.csv')


def parser(url, low_price, top_price, global_df):
    catalog_list = get_catalogs_wb()
    try:
        name_category, shard, query = search_category_in_catalog(url=url, catalog_list=catalog_list)
        data_list = get_content(shard=shard, query=query, low_price=low_price, top_price=top_price)
        save_excel(data_list, f'{name_category}')
        df = pd.DataFrame(data_list)
        df['category'] = name_category
        if global_df.empty:
            global_df = df
        else:
            global_df = global_df.append(df, ignore_index=True)
        return global_df
    except TypeError:
        print('Ошибка! Возможно не верно указан раздел. Удалите все доп фильтры с ссылки')
    except PermissionError:
        print('Ошибка! Вы забыли закрыть созданный ранее excel файл. Закройте и повторите попытку')


data_list = get_catalogs_wb()
categories = [(x['category_name'], x['category_url']) for x in data_list]
random_cats = random.choices(categories, k=5)
print(len(random_cats))
df = pd.DataFrame()

for i in range(len(random_cats)):
    print(random_cats[i][0], random_cats[i][1])
    df = parser(random_cats[i][1], 500, 5000, df)
df.to_csv("products.csv")

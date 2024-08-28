import requests


def get(url: str):
    """

    :param url:
    :return:
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:109.0) Gecko/20100101 Firefox/114.0',
    }
    res = requests.get(url=url, headers=headers)
    return res.text


def post(url: str, paras: dict):
    """

    :param url:
    :param paras:
    :return:
    """
    result = requests.post(url, json=paras)
    return result

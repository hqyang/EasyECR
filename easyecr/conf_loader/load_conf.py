import configparser

from easyecr.common import common_path


def load_conf(filepath):
    """

    :param :
    :return:
    """
    # 创建配置文件对象
    con = configparser.ConfigParser()
    # 读取文件
    con.read(filepath, encoding='utf-8')

    result = {}

    # 获取所有section
    sections = con.sections()
    for section in sections:
        items = con.items(section)
        items = dict(items)
        result[section] = items
    return result


def load_event_extraction_service_conf():
    """

    :return:
    """
    result = load_conf(common_path.project_dir + '/config/event_extraction_service/event_extraction_service.txt')
    return result


if __name__ == '__main__':
    conf = load_event_extraction_service_conf()
    print(conf)

import datetime
from pathlib import Path


def time_change(seconds):
    delta = datetime.timedelta(seconds=seconds)
    # 使用timedelta对象创建一个datetime对象
    base_time = datetime.datetime(1, 1, 1)  # 任意日期，只关心时间部分
    result_time = base_time + delta
    # 将结果格式化为字符串
    formatted_time = result_time.strftime('%H:%M:%S.%f')
    return formatted_time



def write2csv(segments: list, csv_file: Path):
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Name\tStart\tDuration\tTime Format\tType\tDescription\n")
        for each in segments:
            text, start, end, dur = each['text'], each['start'], each['end'], each['duration']
            start_au = time_change(start)
            dur = time_change(end - start)
            f.write(f"{text}\t{start_au}\t{dur}\tdecimal\tCue\t\n")
import os
import json


json_dir = '/data/chenxr/sar/HRSID/HRSID_png/annotations/test/test2017.json'  # json文件路径
out_dir = '/data/chenxr/sar/HRSID/HRSID_png/annotations/test/'  # 输出的 txt 文件路径


def main():
    # 读取 json 文件数据
    with open(json_dir, 'r') as load_f:
        content = json.load(load_f)
        content_img = content['images']
    # 循环处理
    for t in content_img:
        print(t)
        tmp = t['file_name']
        print(tmp)
        filename = out_dir + tmp[0] + '.txt'

        if os.path.exists(filename):
            # 计算 yolo 数据格式所需要的中心点的 相对 x, y 坐标, w,h 的值
            x = (t['bbox'][0] + t['bbox'][2]) / 2 / t['width']
            y = (t['bbox'][1] + t['bbox'][3]) / 2 / t['height']
            w = (t['bbox'][2] - t['bbox'][0]) / t['width']
            h = (t['bbox'][3] - t['bbox'][1]) / t['height']
            fp = open(filename, mode="r+", encoding="utf-8")
            file_str = str(t['category_id']) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + \
                       ' ' + str(round(h, 6))
            line_data = fp.readlines()

            if len(line_data) != 0:
                fp.write('\n' + file_str)
            else:
                fp.write(file_str)
            fp.close()

        # 不存在则创建文件
        else:
            fp = open(filename, mode="w", encoding="utf-8")
            fp.close()


if __name__ == '__main__':
    main()
import pickle
import json


# dump/load pickle
def dump_pickle(data, out_path):
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
        print('write data to', out_path)


def load_pickle(in_path):
    with open(in_path, 'rb') as f:
        data = pickle.load(f)  # list
        return data


# dump/load json
def dump_json(adict, out_path):
    with open(out_path, 'w', encoding='UTF-8') as json_file:
        # 设置缩进，格式化多行保存; ascii False 保存中文
        json_str = json.dumps(adict, indent=2, ensure_ascii=False)
        json_file.write(json_str)


def load_json(in_path):
    with open(in_path, 'rb') as f:
        adict = json.load(f)
        return adict


if __name__ == '__main__':
    a = [1, 2, 3]
    import random

    random.shuffle(a)
    print(a)

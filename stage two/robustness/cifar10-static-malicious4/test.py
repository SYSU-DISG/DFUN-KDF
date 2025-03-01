# 递归函数，用于将嵌套字典中的值取出并添加到列表中
def flatten_dict_values(d, flattened_values):
    for value in d.values():
        if isinstance(value, dict):
            flatten_dict_values(value, flattened_values)
        else:
            flattened_values.append(value)

# 示例嵌套字典
nested_dict = {
    'a': 4,
    'b': {
        'c': [2],
        'd': {
            'e': 3,
            'f': 4
        }
    },
    'g': {
        'h': 5,
        'i': {
            'j': 6
        }
    }
}

# 初始化一个空列表来存放所有值
flattened_values = []

# 调用递归函数将嵌套字典中的值取出并添加到列表中
flatten_dict_values(nested_dict, flattened_values)

# 打印结果
print(flattened_values)

# 示例列表
numbers = [5, 2, 8, 1, 3, 9]

# 对列表进行排序，并获取前两个最小值的索引
sorted_indices = sorted(range(len(numbers)), key=lambda i: numbers[i])[:2]

# 输出最小值的索引
print("Indices of the two smallest numbers:", sorted_indices)

from nltk.corpus import movie_reviews
import hashlib

reviews = []

for fileid in movie_reviews.fileids("pos"):
    reviews.extend(movie_reviews.words(fileid))

for fileid in movie_reviews.fileids("neg"):
    reviews.extend(movie_reviews.words(fileid))

result = 0

### TODO
# 提高精度，采用分组的方式，A组每组B个哈希函数，R记录B个哈希的平均数，最后答案取A个组的中位数
# print("true counts:", len(set(reviews)))  # 39768  log2->15

group = 4
hash = 4  # 每组哈希函数的数量
max_zeros = [[0 for _ in range(hash)] for _ in range(group)]  # 初始化最大值

for i in range(group):
    for j in range(hash):
        # 定义哈希函数

        for word in reviews:
            # 使用 hashlib 计算哈希值
            gen = f"{word}_SYY10158_{i}_NORSHEEP919_{j}"
            # gen = f"{i}_{j}_{word}"
            hasher = hashlib.sha256(gen.encode())
            # hasher.update(word.encode())
            hash_value = int(hasher.hexdigest(), 16)  # 将十六进制转换为整数
            binary_hash = bin(hash_value)  # 将整数转换为二进制字符串
            zero = len(binary_hash) - len(
                binary_hash.rstrip('0'))  # 计算尾部连续零的数量
            max_zeros[i][j] = max(max_zeros[i][j], zero)  # 更新最大值

print(max_zeros)  # 每组的哈希函数的最大值
# 对每个组内的多个哈希函数求平均值
groups_r = [0 for _ in range(group)]
for i in range(group):
    groups_r[i] = sum(max_zeros[i]) / hash  # group可以取小数，更多元（不限制在2的整数幂次）

# 对每个组的平均值取中位数
print(groups_r)
groups_r.sort()
if group % 2 == 0:
    # 偶数，取中间两个的平均数
    max_zero = (groups_r[group // 2] + groups_r[group // 2 - 1]) / 2
else:
    # 奇数，取中间的那个
    max_zero = groups_r[group // 2]

# 取平均数
# max_zero = sum(groups_r) / group  # 取平均数

result = int(2**max_zero)  # 计算最终结果
print("group:", group, "  hash:", hash)

### end of TODO

print(f"{result}")

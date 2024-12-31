import numpy as np
import matplotlib.pyplot as plt
import LVQbits
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import font_manager


font_path = "/Users/austindai/Downloads/SimHei.ttf"  # 宋体路径
font = font_manager.FontProperties(fname=font_path)




#错误版本，不应该把所有query向量一起计算，这样的意义不大
def compute_distances(query, data, chunk_size=1):
    num_query = query.shape[0]
    num_data = data.shape[0]
    distances = np.zeros((num_query, num_data), dtype=np.float32)

    for i in tqdm(range(0, num_query, chunk_size), desc="Computing distances"):
        query_chunk = query[i:i + chunk_size]
        distances_chunk = np.sum((query_chunk[:, None] - data) ** 2, axis=2)
        distances[i:i + chunk_size] = distances_chunk

    return distances

def compute_metrics(distances):
    max_distance = np.max(distances)
    mean_distance = np.mean(distances)
    return max_distance, mean_distance

def compute_distance_ratios(original_distances, compressed_distances):
    epsilon = 1e-10
    ratios = compressed_distances / (original_distances + epsilon)
    return ratios

def plot_results(original_distances, compressed_distances, ratios):
    original_flat = original_distances.flatten()
    compressed_flat = compressed_distances.flatten()

    plt.figure(figsize=(12, 6))

    # 距离分布直方图
    plt.subplot(1, 2, 1)
    plt.hist(original_flat, bins=50, alpha=0.7, label="原始距离", color="blue")
    plt.hist(compressed_flat, bins=50, alpha=0.7, label="压缩距离", color="green")
    plt.xlabel("距离", fontproperties=font)
    plt.ylabel("频率", fontproperties=font)
    plt.title("距离分布", fontproperties=font)
    plt.legend(prop=font)

    # 距离比例直方图
    plt.subplot(1, 2, 2)
    plt.hist(ratios.flatten(), bins=50, color='orange', alpha=0.7, label="距离比例")
    plt.axvline(x=1.0, color='red', linestyle='--')  # 不再传递 fontproperties 参数
    plt.text(1.0, plt.ylim()[1] * 0.95, "理想比例 (1.0)", color='red', fontproperties=font, horizontalalignment='left')
    plt.xlabel("距离比例 (压缩 / 原始)", fontproperties=font)
    plt.ylabel("频率", fontproperties=font)
    plt.title("距离比例分布", fontproperties=font)
    plt.legend(prop=font)

    plt.tight_layout()
    plt.show()





    normalized_squared_error = np.sqrt(np.abs((compressed_flat - original_flat) / original_flat))

    mean_error = np.mean(normalized_squared_error)
    max_error = np.max(normalized_squared_error)
    min_error = np.min(normalized_squared_error)


    print(mean_error)
    print(max_error)
    print(min_error)





def calculate_average_perQueryVector(query_vectors, original_data, compressed_data):
    """
    计算每个 query 向量到原始数据和压缩数据的平均距离增长百分比和最大距离增长百分比，
    然后分别对所有 query 向量的这两个百分比取平均值。
    """
    avg_distance_increases = []  # 平均距离增长百分比
    max_distance_increases = []  # 最大距离增长百分比

    for query in tqdm(query_vectors, desc="目前已经计算到的query", unit="query"):
        # 计算 query 到原始数据集的平方欧式距离
        original_distances = np.sum((original_data - query) ** 2, axis=1)
        original_avg_distance = np.mean(original_distances)  # 平均距离
        original_max_distance = np.max(original_distances)  # 最大距离

        # 计算 query 到压缩数据集的平方欧式距离
        compressed_distances = np.sum((compressed_data - query) ** 2, axis=1)
        compressed_avg_distance = np.mean(compressed_distances)  # 平均距离
        compressed_max_distance = np.max(compressed_distances)  # 最大距离

        # 计算平均距离增长百分比
        avg_increase_percentage = ((compressed_avg_distance - original_avg_distance) / original_avg_distance) * 100
        # 计算最大距离增长百分比
        max_increase_percentage = ((compressed_max_distance - original_max_distance) / original_max_distance) * 100

        # 存储结果
        avg_distance_increases.append(avg_increase_percentage)
        max_distance_increases.append(max_increase_percentage)

    # 分别对所有 query 向量的增长百分比取平均值
    avg_distance_percentage = np.mean(avg_distance_increases)
    max_distance_percentage = np.mean(max_distance_increases)

    return avg_distance_percentage, max_distance_percentage











if __name__ == "__main__":
  query_path = "/Users/austindai/Downloads/gist_query.fvecs"  # Replace with actual query file path
  data_path = "/Users/austindai/Downloads/gist_base.fvecs"  # Replace with actual original data file path

  query = LVQbits.read_Fvecs(query_path)

  original_data = LVQbits.read_Fvecs(data_path)

  compressed_data = LVQbits.LVQ_firstLevel(original_data,1)[0]

  """错误的实验
  original_distances = compute_distances(query, original_data)
  compressed_distances = compute_distances(query, compressed_data)
  print(original_distances.shape)
  print(compressed_distances.shape)
  original_max_distance, original_mean_distance = compute_metrics(original_distances)
  compressed_max_distance, compressed_mean_distance = compute_metrics(compressed_distances)
  ratios = compute_distance_ratios(original_distances, compressed_distances)
  print(f"Original Data - Max Distance: {original_max_distance}, Mean Distance: {original_mean_distance}")
  print(f"Compressed Data - Max Distance: {compressed_max_distance}, Mean Distance: {compressed_mean_distance}")
  plot_results(original_distances, compressed_distances, ratios) 错误的实验
  """

#新的实验，把每个向量计算对应的增大率存为数组，计算数组的最大值和平均值。

# 调用函数
avg_distance_percentage, max_distance_percentage = calculate_average_perQueryVector(query, original_data, compressed_data)

print("平均距离增长百分比的平均值:", avg_distance_percentage)
print("最大距离增长百分比的平均值:", max_distance_percentage)


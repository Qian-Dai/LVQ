import numpy as np
import matplotlib.pyplot as plt
import LVQbits
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import font_manager


font_path = "/Users/austindai/Downloads/SimHei.ttf"  # 宋体路径
font = font_manager.FontProperties(fname=font_path)

def compute_distances(query, data, chunk_size=50):
    num_query = query.shape[0]
    num_data = data.shape[0]
    distances = np.zeros((num_query, num_data), dtype=np.float32)

    for i in tqdm(range(0, num_query, chunk_size), desc="Computing distances"):
        query_chunk = query[i:i + chunk_size]
        distances_chunk = np.linalg.norm(query_chunk[:, None] - data, axis=2)
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





if __name__ == "__main__":
  query_path = "/Users/austindai/Downloads/gist_query.fvecs"  # Replace with actual query file path
  data_path = "/Users/austindai/Downloads/gist_base.fvecs"  # Replace with actual original data file path


  query = LVQbits.read_Fvecs(query_path)[:1000]
  original_data = LVQbits.read_Fvecs(data_path)[:100000]
  compressed_data = LVQbits.LVQ_firstLevel(original_data,1)[0][:100000]

  print("Compressed data type:", type(compressed_data))
  print("Compressed data sample:", compressed_data[:5])






  original_distances = compute_distances(query, original_data)
  compressed_distances = compute_distances(query, compressed_data)


  original_max_distance, original_mean_distance = compute_metrics(original_distances)
  compressed_max_distance, compressed_mean_distance = compute_metrics(compressed_distances)


  ratios = compute_distance_ratios(original_distances, compressed_distances)


  print(f"Original Data - Max Distance: {original_max_distance}, Mean Distance: {original_mean_distance}")
  print(f"Compressed Data - Max Distance: {compressed_max_distance}, Mean Distance: {compressed_mean_distance}")


  plot_results(original_distances, compressed_distances, ratios)
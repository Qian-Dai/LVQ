import numpy as np
import time


def LVQ_firstLevel(Xin: np.ndarray, B1: float) -> (np.ndarray, float, float):

    start_time = time.time()

    # 中心化
    X_mean = np.mean(Xin)
    X = Xin - X_mean

    # 计算量化范围和步长
    u = np.max(X)
    l = np.min(X)
    delta = (u - l) / (2 ** B1 - 1)
    delta = max(delta, 1.0)  # 避免步长为 0

    # 计算量化码和重建数据
    quantized = np.clip(np.floor((X - l) / delta + 0.5), 0, 2 ** B1 - 1)
    reconstructed = quantized * delta + l + X_mean

    # 计算时间
    elapsed_time = time.time() - start_time

    # 计算均方误差
    mse = np.mean((Xin - reconstructed) ** 2)
    return reconstructed, elapsed_time, mse


def LVQ_twoLevel(Xin: np.ndarray, B1: float, B2: float) -> (np.ndarray, float, float):

    start_time = time.time()

    # 一级量化
    X_mean = np.mean(Xin)
    X = Xin - X_mean
    u = np.max(X)
    l = np.min(X)
    delta1 = (u - l) / (2 ** B1 - 1)
    delta1 = max(delta1, 1.0)
    quantized1 = np.clip(np.floor((X - l) / delta1 + 0.5), 0, 2 ** B1 - 1)
    reconstructed1 = quantized1 * delta1 + l

    # 计算残差
    residual = X - reconstructed1

    # 二级量化
    u_res = np.max(residual)
    l_res = np.min(residual)
    delta2 = (u_res - l_res) / (2 ** B2 - 1)
    delta2 = max(delta2, 1.0)
    quantized2 = np.clip(np.floor((residual - l_res) / delta2 + 0.5), -2 ** (B2 - 1), 2 ** (B2 - 1) - 1)
    reconstructed2 = quantized2 * delta2 + l_res

    # 重建数据
    reconstructed = reconstructed1 + reconstructed2 + X_mean

    # 计算时间
    elapsed_time = time.time() - start_time

    # 计算均方误差
    mse = np.mean((Xin - reconstructed) ** 2)
    return reconstructed, elapsed_time, mse


def evaluate_LVQ(data: np.ndarray, B1: float, B2: float) -> None:

    n_samples = data.shape[0]

    # 记录一级量化和二级量化的时间和误差
    time1_list, mse1_list = [], []
    time2_list, mse2_list = [], []

    for i in range(n_samples):
        vector = data[i]

        # 一级量化
        _, time1, mse1 = LVQ_firstLevel(vector, B1)
        time1_list.append(time1)
        mse1_list.append(mse1)

        # 二级量化
        _, time2, mse2 = LVQ_twoLevel(vector, B1, B2)
        time2_list.append(time2)
        mse2_list.append(mse2)

    # 计算平均时间和误差
    avg_time1 = np.mean(time1_list)
    avg_mse1 = np.mean(mse1_list)
    avg_time2 = np.mean(time2_list)
    avg_mse2 = np.mean(mse2_list)

    # 输出结果
    print(f"一级量化平均时间: {avg_time1:.6f} 秒, 平均误差: {avg_mse1:.6f}")
    print(f"二级量化平均时间: {avg_time2:.6f} 秒, 平均误差: {avg_mse2:.6f}")

# 读取 .fvecs 文件中的数据
def read_Fvecs(file_path):

    with open(file_path, 'rb') as f:
        data = f.read()

    vectors = []
    offset = 4  # 跳过文件头部的4字节
    dims = 960  # 修改为你的数据维度
    while offset < len(data):
        vector = np.frombuffer(data, dtype=np.float32, count=dims, offset=offset)
        vectors.append(vector)
        offset += dims * 4 + 4  # 跳过当前向量数据

    return np.vstack(vectors)


# 示例
if __name__ == "__main__":

    # 读取数据集
    data = read_Fvecs('/Users/austindai/Downloads/gist_base.fvecs')  # 使用实际路径
   # 量化比特数
    B1 = 1
    B2 = 1.6

    # 评估一级和二级量化
    evaluate_LVQ(data, B1, B2)
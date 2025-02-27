import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import CubicSpline

plt.rcParams['font.family']='Times New Roman ,SimSun '# 设置字体族，中文为SimSun，英文为Times New Roman

# 读取图像并转换为灰度
image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 定义滑动窗口大小
window_size = 200  # 可以调整窗口大小，如 30, 50, 100
h, w = gray.shape

# CLAHE 自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_clahe = clahe.apply(gray)

adaptive_otsu = np.zeros_like(gray_clahe)  # 创建空的二值化图像

# 滑动窗口方式应用 Otsu
for y in range(0, h, window_size):
    for x in range(0, w, window_size):
        # 计算当前窗口的区域
        y_end = min(y + window_size, h)
        x_end = min(x + window_size, w)
        
        # 获取窗口内的图像
        roi = gray[y:y_end, x:x_end]
        
        # 仅在非空区域应用 Otsu
        if roi.size > 0:
            _, local_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            adaptive_otsu[y:y_end, x:x_end] = local_thresh

# 形态学操作：去除噪声
kernel = np.ones((10,10), np.uint8)  # 结构元素大小可调
morph_cleaned = cv2.morphologyEx(adaptive_otsu, cv2.MORPH_OPEN, kernel, iterations=2)

n = 10000  # 设定最小面积阈值，可根据实际情况调整

# 轮廓检测
contours, _ = cv2.findContours(morph_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个新图像，去除小块
filtered_mask = np.zeros_like(morph_cleaned)

# 遍历轮廓，仅保留面积 >= n 的区域
for cnt in contours:
    if cv2.contourArea(cnt) >= n:
        cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

skeleton = cv2.ximgproc.thinning(filtered_mask)

# 确保是二值图像
_, skeleton_binary = cv2.threshold(skeleton, 127, 255, cv2.THRESH_BINARY)

# 查找所有骨架的连通部分
contours, _ = cv2.findContours(skeleton_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 创建一个新图像，存放最终结果
final_skeleton = np.zeros_like(skeleton_binary)

# 计算端点间欧几里得距离
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 设定U形判断阈值
U_SHAPE_RATIO_THRESHOLD = 1.5  # 可调整
MIN_U_SHAPE_LENGTH = 50  # 设定最小 U 形骨架长度

# 处理每个独立的骨架结构
for cnt in contours:
    if len(cnt) > 10:  # 过滤太短的曲线
        # 转换为点列表
        points = cnt[:, 0, :]  # (N, 2) 数组
        
        # 构建一个图，点与点之间建立边
        G = nx.Graph()
        for i in range(len(points) - 1):
            p1, p2 = tuple(points[i]), tuple(points[i + 1])
            G.add_edge(p1, p2, weight=np.linalg.norm(np.array(p1) - np.array(p2)))

        # 找到骨架上的端点（度数为1的点）
        endpoints = [node for node in G.nodes if G.degree(node) == 1]

        # 计算最长路径
        longest_path = []
        max_length = 0
        for start in endpoints:
            for end in endpoints:
                if start != end:
                    path = nx.shortest_path(G, source=start, target=end, weight='weight')
                    length = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                    if length > max_length:
                        max_length = length
                        longest_path = path
        # 计算端点间直线距离
        if longest_path:
            start, end = longest_path[0], longest_path[-1]
            direct_distance = euclidean_distance(start, end)
            # 判断是否为 U 形结构
            if direct_distance > 0 and max_length / direct_distance > U_SHAPE_RATIO_THRESHOLD and max_length > MIN_U_SHAPE_LENGTH:
                continue  # 直接跳过，不加入最终结果

        # 绘制正常骨架
        for i in range(len(longest_path) - 1):
            cv2.line(final_skeleton, longest_path[i], longest_path[i + 1], 255, 1)


# 查找所有骨架的连通部分
filtered_contours, _ = cv2.findContours(final_skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 创建一个新图像存放二次曲线拟合结果
quadratic_fitted_skeleton = np.zeros_like(final_skeleton)

# 设定 MSE 误差阈值（可调）
error_threshold = 100  # 你可以调整这个值，数值越小，筛选越严格

# 存储最小曲率半径点及其数值
min_curvature_points = []
for cnt in filtered_contours:
    points = cnt[:, 0, :]  # 获取 (N, 2) 坐标点
    x_vals, y_vals = points[:, 0], points[:, 1]

    if len(x_vals) > 3:  # 至少 3 个点才能进行二次拟合
        # 使用二次多项式拟合
        coefficients = np.polyfit(x_vals, y_vals, 2)
        poly_func = np.poly1d(coefficients)

        # 计算拟合误差 (MSE)
        y_predicted = poly_func(x_vals)  # 计算拟合曲线的 y 值
        mse = np.mean((y_vals - y_predicted) ** 2)  # 计算均方误差
        # 仅保留 MSE 小于阈值的曲线
        if mse < error_threshold:
            if cnt is not None:
                for point in cnt:
                    x, y = point[0]
                    cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)  # 用蓝色点绘制
            # 生成平滑曲线点
            x_fitted = np.linspace(min(x_vals), max(x_vals), 100).astype(int)
            y_fitted = poly_func(x_fitted).astype(int)

            # 去掉曲线前后 10 个点，防止边界问题
            trimmed_x_fitted = x_fitted[10:-10]
            trimmed_y_fitted = y_fitted[10:-10]

            # **计算曲率半径**
            dx = np.polyder(poly_func, 1)(trimmed_x_fitted)  # 一阶导数 dy/dx
            ddx = np.polyder(poly_func, 2)(trimmed_x_fitted)  # 二阶导数 d²y/dx²

            # 避免除零错误
            ddx[ddx == 0] = 1e-6

            curvature_radius = (1 + dx ** 2) ** (3 / 2) / np.abs(ddx)

            # 找到曲率最小的点
            min_curvature_idx = np.argmin(curvature_radius)
            min_curvature_x = trimmed_x_fitted[min_curvature_idx]
            min_curvature_y = trimmed_y_fitted[min_curvature_idx]
            min_curvature_value = curvature_radius[min_curvature_idx]

            # 记录最小曲率点
            min_curvature_points.append((min_curvature_x, min_curvature_y, min_curvature_value))

            # 绘制黄色拟合曲线
            valid_points = [(x, y) for x, y in zip(x_fitted, y_fitted) if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]]
            if len(valid_points) > 1:
                cv2.polylines(image, [np.array(valid_points)], isClosed=False, color=(0, 255, 255), thickness=3)  # 黄色

# 绘制曲率最小的点
min_curvature_global = min(min_curvature_points, key=lambda p: p[2])  # p[2] 是曲率值

min_curvature_x, min_curvature_y, min_curvature_value = min_curvature_global
if 0 <= min_curvature_x < image.shape[1] and 0 <= min_curvature_y < image.shape[0]:  # 确保点在图像范围内
    cv2.drawMarker(image, (min_curvature_x, min_curvature_y), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=5)  # 红色十字
    cv2.putText(image, f"{min_curvature_value:.1f}", (min_curvature_x + 5, min_curvature_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # 标注数值

# 显示最终结果
# plt.figure(figsize=(12, 6))

# plt.subplot(2, 3, 1)
# plt.title("二值化")
# plt.imshow(adaptive_otsu, cmap="gray")
# plt.axis("off")

# plt.subplot(2, 3, 2)
# plt.title("开运算")
# plt.imshow(morph_cleaned, cmap="gray")
# plt.axis("off")

# plt.subplot(2, 3, 3)
# plt.title("区域筛选")
# plt.imshow(filtered_mask, cmap="gray")
# plt.axis("off")

# plt.subplot(2, 3, 4)
# plt.title("骨架")
# plt.imshow(skeleton, cmap="gray")
# plt.axis("off")

# plt.subplot(2, 3, 5)
# plt.title("骨架筛选")
# plt.imshow(final_skeleton, cmap="gray")
# plt.axis("off")

# plt.subplot(2, 3, 6)
# plt.title("拟合")
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.plot([], [], 'g-', linewidth=3, label="拟合骨架")
# plt.plot([], [], 'y-', linewidth=3, label="拟合曲线")
# plt.plot([], [], 'rx', markersize=10, label="最小曲率点")
# plt.legend()

# plt.show()



fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image, cmap="gray")

ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax.axis("off")
ax.plot([], [], 'g-', linewidth=3, label="拟合骨架")
ax.plot([], [], 'y-', linewidth=3, label="拟合曲线")
ax.plot([], [], 'rx', markersize=10, label="最小曲率点")
ax.legend()

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.show()
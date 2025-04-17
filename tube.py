import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import splprep, splev

plt.rcParams['font.family']='Times New Roman ,SimSun '# 设置字体族，中文为SimSun，英文为Times New Roman
# 可调整参数
WINDOW_SIZE = 600  # 滑动窗口大小
CLAHE_CLIP_LIMIT = 2.0  # CLAHE 对比度限制
CLAHE_GRID_SIZE = (8, 8)  # CLAHE 网格大小
MORPH_KERNEL_SIZE = (10, 10)  # 形态学开运算核大小
MIN_AREA_THRESHOLD = 10000  # 轮廓最小面积阈值
U_SHAPE_RATIO_THRESHOLD = 1.5  # U 形结构判定阈值
MIN_U_SHAPE_LENGTH = 50  # U 形结构最小长度
CURVATURE_ERROR_THRESHOLD = 100  # 曲率拟合误差阈值


def read_and_preprocess_image(image_path):
    """读取图像并转换为灰度"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def apply_adaptive_threshold(gray):
    """应用 CLAHE 进行对比度增强，并使用滑动窗口 Otsu 二值化"""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    gray_clahe = clahe.apply(gray)

    h, w = gray.shape
    binary_image = np.zeros_like(gray_clahe)

    for y in range(0, h, WINDOW_SIZE):
        for x in range(0, w, WINDOW_SIZE):
            y_end, x_end = min(y + WINDOW_SIZE, h), min(x + WINDOW_SIZE, w)
            roi = gray[y:y_end, x:x_end]

            if roi.size > 0:
                _, local_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                binary_image[y:y_end, x:x_end] = local_thresh

    return binary_image


def morphological_cleaning(binary_image):
    """应用形态学开运算去除噪声"""
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)


def remove_small_objects(cleaned_image):
    """去除面积小于阈值的区域"""
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(cleaned_image)

    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_AREA_THRESHOLD:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return filtered_mask


def extract_skeleton(filtered_mask):
    """提取图像骨架"""
    skeleton = cv2.ximgproc.thinning(filtered_mask)
    return cv2.threshold(skeleton, 127, 255, cv2.THRESH_BINARY)[1]


def filter_skeleton(skeleton_binary):
    """筛选骨架，去除 U 形结构"""
    contours, _ = cv2.findContours(skeleton_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    final_skeleton = np.zeros_like(skeleton_binary)

    def euclidean_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    for cnt in contours:
        if len(cnt) > 10:
            points = cnt[:, 0, :]
            G = nx.Graph()
            for i in range(len(points) - 1):
                G.add_edge(tuple(points[i]), tuple(points[i + 1]), weight=euclidean_distance(points[i], points[i + 1]))

            endpoints = [node for node in G.nodes if G.degree(node) == 1]
            longest_path = max(
                (nx.shortest_path(G, source=s, target=e, weight='weight') for s in endpoints for e in endpoints if s != e),
                key=lambda path: sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:])),
                default=[],
            )

            if longest_path:
                start, end = longest_path[0], longest_path[-1]
                direct_distance = euclidean_distance(start, end)
                path_length = sum(G[u][v]['weight'] for u, v in zip(longest_path[:-1], longest_path[1:]))

                if direct_distance > 0 and path_length / direct_distance > U_SHAPE_RATIO_THRESHOLD and path_length > MIN_U_SHAPE_LENGTH:
                    continue

                for i in range(len(longest_path) - 1):
                    cv2.line(final_skeleton, longest_path[i], longest_path[i + 1], 255, 1)

    return final_skeleton


def fit_curves(final_skeleton, image):
    """ 使用 B 样条平滑骨架曲线并计算曲率 """
    contours, _ = cv2.findContours(final_skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_curvature_points = []

    for cnt in contours:
        points = cnt[:, 0, :]
        if len(points) < 5:  # 过少点无法平滑
            continue

        # 提取 x, y 坐标
        x_vals, y_vals = points[:, 0], points[:, 1]

        # B 样条拟合
        tck, u = splprep([x_vals, y_vals], s=10)  # s 控制平滑度，值越大越光滑
        u_fine = np.linspace(0, 1, 100)  # 生成均匀间隔的参数
        x_smooth, y_smooth = splev(u_fine, tck)

        # 计算一阶导数（dx/ds, dy/ds）
        dx, dy = splev(u_fine, tck, der=1)

        # 计算二阶导数（d²x/ds², d²y/ds²）
        ddx, ddy = splev(u_fine, tck, der=2)

        # 计算曲率 k = (dx * ddy - dy * ddx) / (dx² + dy²)^(3/2)
        denominator = (dx**2 + dy**2) ** (3/2)
        denominator[denominator == 0] = 1e-6  # 避免除零
        curvature = np.abs(dx * ddy - dy * ddx) / denominator

        # 找到最大曲率点
        max_curvature_idx = np.argmax(curvature)
        max_curvature_x, max_curvature_y = int(x_smooth[max_curvature_idx]), int(y_smooth[max_curvature_idx])
        max_curvature_value = curvature[max_curvature_idx]

        # 记录最大曲率点
        max_curvature_points.append((max_curvature_x, max_curvature_y, max_curvature_value))

        # 绘制平滑后的曲线
        for i in range(len(x_smooth) - 1):
            cv2.line(image, (int(x_smooth[i]), int(y_smooth[i])), 
                    (int(x_smooth[i+1]), int(y_smooth[i+1])), (0, 255, 255), 2)

    return max(max_curvature_points, key=lambda p: p[2]) if max_curvature_points else None


import matplotlib.lines as mlines

def visualize_results(image, final_skeleton, max_curvature_point):
    """可视化最终结果，包括骨架、拟合曲线和最大曲率点"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 标注最大曲率点
    if max_curvature_point:
        x, y, curvature = max_curvature_point
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # 确保点在图像范围内
            print("???")
            cv2.drawMarker(image, (x, y), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=5)  # 红色十字
            cv2.putText(image, f"{curvature:.1f}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # 标注数值

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 创建图例句柄
    bone_legend = mlines.Line2D([], [], color='green', lw=3, label="骨架")
    fit_curve_legend = mlines.Line2D([], [], color='yellow', lw=3, label="拟合曲线")
    curvature_legend = mlines.Line2D([], [], marker='x', color='red', markersize=10, lw=0, label="最大曲率点")

    # 添加图例（确保参数格式正确）
    ax.legend(handles=[bone_legend, fit_curve_legend, curvature_legend])

    # 去除坐标轴
    ax.axis("off")

    plt.show()




if __name__ == "__main__":
    image, gray = read_and_preprocess_image("image.jpg")
    binary_image = apply_adaptive_threshold(gray)
    cleaned_image = morphological_cleaning(binary_image)
    filtered_mask = remove_small_objects(cleaned_image)
    skeleton = extract_skeleton(filtered_mask)
    final_skeleton = filter_skeleton(skeleton)
    max_curvature_point = fit_curves(final_skeleton, image)
    visualize_results(image, final_skeleton, max_curvature_point)

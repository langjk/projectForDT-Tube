import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 读取图像
image_path = "image.jpg"  # 请替换为你的图片路径
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ==================== 方法: 多通道梯度边缘检测 ====================
def multi_channel_edge_detection(image):
    """ 基于多通道梯度的边缘检测 """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    channels = [image, lab, hsv]
    edges = np.zeros(image.shape[:2], dtype=np.uint8)

    for channel in channels:
        for i in range(3):  # 分别处理每个通道
            grad_x = cv2.Sobel(channel[:, :, i], cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(channel[:, :, i], cv2.CV_64F, 0, 1, ksize=3)

            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

            edges = cv2.bitwise_or(edges, gradient_magnitude)  # 融合多个通道的梯度信息

    _, edges_binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    return edges_binary

# ==================== 形态学操作（闭运算填充孔洞） ====================
def apply_closing(edges):
    """ 进行闭运算，填充小孔洞 """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# ==================== 交互式点击选择目标区域 ====================
selected_mask = None
original_size = image.shape[:2]  # 记录原始图像尺寸

# 设置缩放后的窗口大小（适应屏幕）
display_size = (800, 600)
scale_x = original_size[1] / display_size[0]  # X 轴缩放比例
scale_y = original_size[0] / display_size[1]  # Y 轴缩放比例

def select_region(event, x, y, flags, param):
    global selected_mask

    if event == cv2.EVENT_LBUTTONDOWN:
        # 将点击坐标转换回原始图像坐标
        x_orig = int(x * scale_x)
        y_orig = int(y * scale_y)

        for i, contour in enumerate(contours):
            if cv2.pointPolygonTest(contour, (x_orig, y_orig), False) >= 0:
                selected_mask = np.zeros_like(edges)
                cv2.drawContours(selected_mask, [contours[i]], -1, 255, thickness=cv2.FILLED)
                cv2.imshow("Selected Region", cv2.resize(selected_mask, display_size))  # 显示选区
                print(f"✅ 选中区域 {i}，按 ESC 退出")

# 计算多通道边缘
edges = multi_channel_edge_detection(image)

# 形态学处理（闭运算）
edges = apply_closing(edges)

# 轮廓检测
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 调整窗口大小，但不影响数据
resized_edges = cv2.resize(edges, display_size)

# 显示可点击的边缘图
cv2.imshow("Click to Select Pipeline", resized_edges)
cv2.setMouseCallback("Click to Select Pipeline", select_region)

# 等待用户交互
print("🖱️ 点击目标管道区域...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果用户选择了区域，显示最终结果
if selected_mask is not None:
    final_result = cv2.bitwise_and(image_rgb, image_rgb, mask=selected_mask)

    # 显示最终结果
    plt.figure(figsize=(8, 6))
    plt.imshow(final_result)
    plt.axis("off")
    plt.title("Final Selected Pipeline")
    plt.show()
else:
    print("⚠️ 未选择任何区域，请重新运行程序并点击目标区域。")

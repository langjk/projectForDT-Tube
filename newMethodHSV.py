import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

plt.rcParams['font.family'] = 'Times New Roman, SimSun'

# ==================== 参数配置区 ====================
class Config:
    # 预处理参数
    CLAHE_CLIP_LIMIT = 3.0         # CLAHE对比度限制
    CLAHE_GRID_SIZE = (8, 8)       # CLAHE网格尺寸
    
    # HSV颜色阈值 (需根据实际管道颜色调整)
    HSV_LOWER = [20, 50, 50]       # HSV下限 (H, S, V)
    HSV_UPPER = [40, 255, 255]     # HSV上限
    
    # 形态学参数
    MORPH_CLOSE_SIZE = (15, 15)    # 闭运算核尺寸（填充孔洞）
    MORPH_OPEN_SIZE = (5, 5)       # 开运算核尺寸（去除噪点）
    
    # 骨架处理参数
    MIN_CONTOUR_AREA = 5000        # 最小轮廓面积阈值
    SKELETON_THICKNESS = 3         # 骨架显示粗细
    
    # 曲率计算参数
    SMOOTHNESS = 1.5               # 曲线平滑系数（越大越平滑）
    CURVATURE_SCALE = 1000         # 曲率显示缩放系数

# ==================== 核心函数 ====================
def pipeline(img_path, config=Config):
    """完整处理流程"""
    # 1. 读取图像
    img = cv2.imread(img_path)
    img_preprocessed = clahe_preprocess(img, config)

    # 2. HSV颜色分割
    mask = hsv_segmentation(img_preprocessed, config)

    # 3. 形态学优化
    mask_clean = morphological_optimization(mask, config)

    # 4. 提取轮廓
    main_contour = find_main_contour(mask_clean, config)
    if main_contour is None:
        print("⚠️ 无法找到管道轮廓，可能需要调整 HSV 颜色范围")
        return

    # 5. 骨架提取
    skeleton = extract_refined_skeleton(main_contour, img.shape)

    # 6. 计算曲率
    max_curvature = calculate_curvature(skeleton, img, config)

    # 7. 可视化
    visualize_results(img, skeleton, max_curvature, config)


# ==================== 子函数实现 ====================
def clahe_preprocess(img, config):
    """自适应光照预处理"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_GRID_SIZE
    )
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

def hsv_segmentation(img, config):
    """HSV颜色空间分割"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(config.HSV_LOWER, dtype=np.uint8)
    upper = np.array(config.HSV_UPPER, dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def morphological_optimization(mask, config):
    """形态学优化掩膜"""
    # 闭运算填充孔洞
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        config.MORPH_CLOSE_SIZE
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # 开运算去除噪点
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        config.MORPH_OPEN_SIZE
    )
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

def find_main_contour(mask, config):
    """寻找最大连通域"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > config.MIN_CONTOUR_AREA]

    if not contours:
        print("⚠️ 没找到合适的管道轮廓，可能是颜色范围不正确或者背景干扰")
        return None
    
    return max(contours, key=cv2.contourArea)


def extract_refined_skeleton(contour, img_shape):
    """ 精确骨架提取 """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # 先细化管道区域 -> 得到骨架
    skeleton = cv2.ximgproc.thinning(mask)
    
    return skeleton

def calculate_curvature(skeleton, img, config):
    """B样条曲率计算"""
    points = np.column_stack(np.where(skeleton > 10))
    if len(points) < 10:
        return None

    try:
        # B样条拟合
        tck, u = splprep(points.T, u=None, s=config.SMOOTHNESS)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck)
        
        # 计算导数
        dx, dy = splev(u_new, tck, der=1)
        ddx, ddy = splev(u_new, tck, der=2)

        # 计算曲率
        denominator = np.power(dx**2 + dy**2, 1.5)
        denominator[denominator == 0] = 1e-6  # 避免除零错误
        curvature = np.abs(dx*ddy - dy*ddx) / denominator

        # 防止 `nan`
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

        # 找到最大曲率点
        max_idx = np.argmax(curvature)
        max_point = (int(x_new[max_idx]), int(y_new[max_idx]))
        max_value = curvature[max_idx] * config.CURVATURE_SCALE

        # 绘制曲线
        for i in range(len(x_new)-1):
            cv2.line(img, 
                    (int(x_new[i]), int(y_new[i])),
                    (int(x_new[i+1]), int(y_new[i+1])),
                    (0, 255, 255), config.SKELETON_THICKNESS)
        
        return (max_point[0], max_point[1], max_value)
    
    except Exception as e:
        print(f"曲率计算失败: {e}")
        return None


def visualize_results(img, skeleton, max_curvature, config):
    """可视化结果"""
    plt.figure(figsize=(12, 6))
    
    # 原图与分割结果
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image with Detection')
    
    # 标注最大曲率点
    if max_curvature:
        x, y, k = max_curvature
        plt.scatter(x, y, s=200, c='red', marker='x', linewidths=3)
        plt.text(x+20, y-20, f"Curvature: {k:.2f}", 
                fontsize=12, color='white', 
                bbox=dict(facecolor='red', alpha=0.8))
    
    # 显示骨架
    plt.subplot(122)
    plt.imshow(skeleton, cmap='gray')
    plt.title('Refined Skeleton')
    
    plt.tight_layout()
    plt.show()

# 通过以下代码交互式调整HSV阈值
def adjust_hsv(img_path):
    img = cv2.imread(img_path)
    cv2.namedWindow('HSV Adjust')
    
    # 创建滑动条
    cv2.createTrackbar('H Min', 'HSV Adjust', 0, 179, lambda x: None)
    cv2.createTrackbar('H Max', 'HSV Adjust', 179, 179, lambda x: None)
    cv2.createTrackbar('S Min', 'HSV Adjust', 0, 255, lambda x: None)
    cv2.createTrackbar('S Max', 'HSV Adjust', 255, 255, lambda x: None)
    cv2.createTrackbar('V Min', 'HSV Adjust', 0, 255, lambda x: None)
    cv2.createTrackbar('V Max', 'HSV Adjust', 255, 255, lambda x: None)

    while True:
        h_min = cv2.getTrackbarPos('H Min', 'HSV Adjust')
        h_max = cv2.getTrackbarPos('H Max', 'HSV Adjust')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Adjust')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Adjust')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Adjust')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Adjust')

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower, upper)
        cv2.imshow('Mask Preview', mask)
        
        if cv2.waitKey(1) == 27:  # ESC退出
            break

    cv2.destroyAllWindows()
    return lower, upper

# ==================== 执行入口 ====================
if __name__ == "__main__":
    try:
        HSV_LOWER,HSV_UPPER = adjust_hsv("image.jpg")
        pipeline("image.jpg")
    except Exception as e:
        print(f"处理出错: {str(e)}")
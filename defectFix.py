import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

def extract_edges(binary_img):
    height, width = binary_img.shape
    white_to_black_edges = []
    black_to_white_edges = []

    for y in range(height):
        row = binary_img[y]
        for x in range(1, width):
            if row[x] == 0 and row[x-1] == 255:  
                white_to_black_edges.append((y, x))
            if row[x] == 255 and row[x-1] == 0:  
                black_to_white_edges.append((y, x))

    return white_to_black_edges, black_to_white_edges

def detect_and_fill_edges(binary_img):
    height, width = binary_img.shape
    for y in range(height):
        row = binary_img[y]
        white_to_black_edges = []
        black_to_white_edges = []

        for x in range(1, width):
            if row[x] == 0 and row[x-1] == 255:  
                white_to_black_edges.append(x)
            if row[x] == 255 and row[x-1] == 0:  
                black_to_white_edges.append(x)

        if len(white_to_black_edges) > 0 and len(black_to_white_edges) > 0:
            left_white_to_black = min(white_to_black_edges)
            right_black_to_white = max(black_to_white_edges)
            if left_white_to_black < right_black_to_white:
                binary_img[y, left_white_to_black:right_black_to_white] = 0 

    return binary_img

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_img = binary_img.astype(np.uint8)

    binary_img[:2, :] = 255  
    binary_img[-2:, :] = 255  
    binary_img[:, :2] = 255  
    binary_img[:, -2:] = 255  

    binary_img = detect_and_fill_edges(binary_img)
    binary_img = binary_img.T  
    binary_img = detect_and_fill_edges(binary_img)
    binary_img = binary_img.T

    white_to_black_edges, black_to_white_edges = extract_edges(binary_img)
    midpoints = [((y1 + y2) // 2, (x1 + x2) // 2) for (y1, x1), (y2, x2) in zip(white_to_black_edges, black_to_white_edges)]

    y_coords = [point[0] for point in midpoints]
    x_coords = [point[1] for point in midpoints]

    p = Polynomial.fit(y_coords, x_coords, 2)
    fitted_x = p(y_coords)

    marked_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for y, x in zip(y_coords, fitted_x):
        cv2.circle(marked_img, (int(x), y), 2, (0, 255, 255), -1)

    return marked_img

def display_result(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

image_path = 'defect.png'
result_img = process_image(image_path)
display_result(result_img)

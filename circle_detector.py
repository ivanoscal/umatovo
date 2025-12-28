"""
Circle Detector Module
Использует OpenCV для распознавания и подсчёта круглых объектов
"""

import cv2
import numpy as np


class CircleDetector:
    """Детектор круглых объектов на изображении"""
    
    def __init__(self):
        pass
    
    def detect_circles(self, image):
        """
        Детектирование кругов методом контурного анализа
        
        Returns:
            tuple: (count, circles_list, output_image)
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Бинаризация для выделения тёмных отверстий
        _, thresh = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
        
        # Морфологическая очистка
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Поиск контуров
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        min_y = int(height * 0.10)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 60 or area > 3500:
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            
            circle_area = np.pi * radius ** 2
            fill_ratio = area / circle_area if circle_area > 0 else 0
            
            if y < min_y or radius < 4 or radius > 38:
                continue
            
            if circularity > 0.35 or fill_ratio > 0.45:
                circles.append((int(x), int(y), int(radius), circularity))
        
        # Удаление дубликатов
        if circles:
            radii = [c[2] for c in circles]
            median_r = np.median(radii)
            circles = self._remove_duplicates(circles, median_r * 1.4)
        
        # Сортировка по рядам
        circles_sorted = sorted(circles, key=lambda c: (c[1]//18, c[0]))
        
        # Создаём выходное изображение
        output = image.copy()
        result_circles = []
        
        for i, (x, y, r, circ) in enumerate(circles_sorted, 1):
            result_circles.append((x, y, r))
            self._draw_circle(output, x, y, r, i)
        
        return len(result_circles), result_circles, output
    
    def _remove_duplicates(self, circles, min_dist):
        """Удаление дубликатов"""
        kept = []
        for c in sorted(circles, key=lambda x: -x[3]):
            is_dup = False
            for k in kept:
                if np.sqrt((c[0]-k[0])**2 + (c[1]-k[1])**2) < min_dist:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(c)
        return kept
    
    def _draw_circle(self, image, x, y, r, number):
        """Отрисовка круга с номером"""
        # Зелёный контур
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        
        # Оранжевый кружок с номером
        cv2.circle(image, (x, y), 9, (0, 140, 255), -1)
        
        # Номер белым цветом
        label = str(number)
        font_scale = 0.32
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(image, label, (x - tw//2, y + th//2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    def add_manual_circle(self, image, x, y, radius, number):
        """Добавление круга вручную"""
        self._draw_circle(image, x, y, radius, number)
        return image


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        detector = CircleDetector()
        count, circles, result = detector.detect_circles(cv2.imread(sys.argv[1]))
        print(f"Найдено: {count}")
        cv2.imwrite("result.jpg", result)

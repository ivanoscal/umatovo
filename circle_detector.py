"""
Circle Detector Module - Универсальный детектор круглых объектов
"""

import cv2
import numpy as np


class CircleDetector:
    """Детектор круглых объектов с адаптивными параметрами"""
    
    def __init__(self):
        pass
    
    def detect_circles(self, image):
        """
        Универсальное детектирование кругов
        Комбинирует несколько методов для лучшего результата
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        all_circles = []
        
        # Метод 1: Автоматический порог Otsu
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        circles1 = self._find_circles_from_thresh(thresh1, height)
        all_circles.extend(circles1)
        
        # Метод 2: Адаптивный порог
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 21, 5)
        circles2 = self._find_circles_from_thresh(thresh2, height)
        all_circles.extend(circles2)
        
        # Метод 3: Несколько фиксированных порогов
        for thresh_val in [60, 80, 100, 120]:
            _, thresh3 = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            circles3 = self._find_circles_from_thresh(thresh3, height)
            all_circles.extend(circles3)
        
        # Метод 4: HoughCircles на улучшенном изображении
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
        
        hough_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                                          minDist=15, param1=70, param2=30,
                                          minRadius=5, maxRadius=100)
        if hough_circles is not None:
            for c in hough_circles[0]:
                if c[1] > height * 0.08:
                    all_circles.append((int(c[0]), int(c[1]), int(c[2]), 0.8))
        
        # Объединяем и фильтруем дубликаты
        if not all_circles:
            return 0, [], image.copy()
        
        # Удаляем дубликаты
        filtered = self._remove_duplicates(all_circles)
        
        # Фильтруем по радиусу (убираем выбросы)
        if len(filtered) > 3:
            radii = [c[2] for c in filtered]
            median_r = np.median(radii)
            filtered = [c for c in filtered if 0.3 * median_r < c[2] < 2.5 * median_r]
        
        # Сортируем по рядам
        circles_sorted = sorted(filtered, key=lambda c: (c[1]//15, c[0]))
        
        # Рисуем результат
        output = image.copy()
        result_circles = []
        
        for i, c in enumerate(circles_sorted, 1):
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            result_circles.append((x, y, r))
            self._draw_circle(output, x, y, r, i)
        
        return len(result_circles), result_circles, output
    
    def _find_circles_from_thresh(self, thresh, img_height):
        """Поиск кругов в бинарном изображении"""
        circles = []
        
        # Морфология
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Контуры
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_y = int(img_height * 0.05)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 50000:
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            
            if radius < 3 or radius > 150:
                continue
            
            if y < min_y:
                continue
            
            # Проверяем заполненность
            circle_area = np.pi * radius ** 2
            fill_ratio = area / circle_area if circle_area > 0 else 0
            
            # Принимаем если достаточно круглый или хорошо заполнен
            if circularity > 0.4 or fill_ratio > 0.5:
                circles.append((int(x), int(y), int(radius), circularity))
        
        return circles
    
    def _remove_duplicates(self, circles):
        """Удаление дубликатов"""
        if not circles:
            return []
        
        # Вычисляем медианный радиус для определения минимального расстояния
        radii = [c[2] for c in circles]
        median_r = np.median(radii)
        min_dist = median_r * 1.2
        
        kept = []
        # Сортируем по круглости (лучшие первыми)
        sorted_circles = sorted(circles, key=lambda c: -c[3])
        
        for c in sorted_circles:
            x, y = c[0], c[1]
            is_dup = False
            for k in kept:
                dist = np.sqrt((x - k[0])**2 + (y - k[1])**2)
                if dist < min_dist:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(c)
        
        return kept
    
    def _draw_circle(self, image, x, y, r, number):
        """Отрисовка круга"""
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 9, (0, 140, 255), -1)
        
        label = str(number)
        font_scale = 0.32
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(image, label, (x - tw//2, y + th//2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        detector = CircleDetector()
        count, circles, result = detector.detect_circles(cv2.imread(sys.argv[1]))
        print(f"Найдено: {count}")
        cv2.imwrite("result.jpg", result)

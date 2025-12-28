"""
Circle Detector - Универсальный детектор с приоритетом на точность
"""

import cv2
import numpy as np


class CircleDetector:
    
    def detect_circles(self, image):
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Определяем зону контента
        row_means = np.mean(gray, axis=1)
        non_black = np.where(row_means > 20)[0]
        min_y = non_black[0] + 5 if len(non_black) > 0 else 0
        max_y = non_black[-1] - 5 if len(non_black) > 0 else height
        
        all_circles = []
        
        # Несколько порогов бинаризации
        for thresh_val in [50, 70, 90, 110, 130]:
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            circles = self._extract_circles(thresh, min_y, max_y)
            all_circles.extend(circles)
        
        # Otsu
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        circles_otsu = self._extract_circles(thresh_otsu, min_y, max_y)
        all_circles.extend(circles_otsu)
        
        if not all_circles:
            return 0, [], image.copy()
        
        # Удаляем дубликаты
        filtered = self._remove_duplicates(all_circles)
        
        # Фильтр по радиусу
        if len(filtered) > 3:
            radii = [c[2] for c in filtered]
            median_r = np.median(radii)
            filtered = [c for c in filtered if 0.5 * median_r < c[2] < 2.0 * median_r]
        
        # Сортировка
        circles_sorted = sorted(filtered, key=lambda c: (c[1]//15, c[0]))
        
        # Рисуем
        output = image.copy()
        result_circles = []
        
        for i, c in enumerate(circles_sorted, 1):
            x, y, r = c[0], c[1], c[2]
            result_circles.append((x, y, r))
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 9, (0, 140, 255), -1)
            label = str(i)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            cv2.putText(output, label, (x - tw//2, y + th//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        
        return len(result_circles), result_circles, output
    
    def _extract_circles(self, thresh, min_y, max_y):
        circles = []
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            
            if y < min_y or y > max_y or radius < 5:
                continue
            
            circle_area = np.pi * radius ** 2
            fill_ratio = area / circle_area
            
            # Качественный круг: хорошая круглость И заполненность
            quality = circularity * fill_ratio
            
            if circularity > 0.45 and fill_ratio > 0.45:
                circles.append((int(x), int(y), int(radius), quality))
        
        return circles
    
    def _remove_duplicates(self, circles):
        if not circles:
            return []
        
        radii = [c[2] for c in circles]
        median_r = np.median(radii)
        min_dist = median_r * 1.2  # Уменьшили для мелких объектов
        
        kept = []
        for c in sorted(circles, key=lambda x: -x[3]):
            x, y = c[0], c[1]
            if not any(np.sqrt((x-k[0])**2 + (y-k[1])**2) < min_dist for k in kept):
                kept.append(c)
        
        return kept


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        d = CircleDetector()
        count, _, result = d.detect_circles(cv2.imread(sys.argv[1]))
        print(f"Найдено: {count}")
        cv2.imwrite("result.jpg", result)

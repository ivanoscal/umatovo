"""
Circle Detector Module
Использует OpenCV для распознавания и подсчёта круглых объектов на изображении
"""

import cv2
import numpy as np


class CircleDetector:
    """Класс для детектирования круглых объектов на изображении"""
    
    def __init__(self):
        """Инициализация детектора кругов с оптимальными параметрами"""
        pass
    
    def detect_circles(self, image):
        """
        Детектирование кругов на изображении
        
        Args:
            image: Изображение в формате numpy array (BGR)
            
        Returns:
            tuple: (количество_кругов, список_кругов, обработанное_изображение)
        """
        # Создаём копию для отрисовки
        output = image.copy()
        height, width = image.shape[:2]
        
        # Конвертируем в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Размытие для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Адаптивные параметры в зависимости от размера изображения
        min_dimension = min(width, height)
        
        # Минимальное расстояние между центрами (5% от меньшей стороны)
        min_dist = max(20, int(min_dimension * 0.03))
        
        # Минимальный и максимальный радиус
        min_radius = max(10, int(min_dimension * 0.01))
        max_radius = int(min_dimension * 0.15)
        
        # Детектируем круги с строгими параметрами
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=100,  # Высокий порог Canny
            param2=50,   # Строгий порог аккумулятора (меньше ложных срабатываний)
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        detected_circles = []
        count = 0
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for i, circle in enumerate(circles[0, :], 1):
                x, y, radius = int(circle[0]), int(circle[1]), int(circle[2])
                detected_circles.append((x, y, radius))
                
                # Зелёный контур вокруг круга
                cv2.circle(output, (x, y), radius, (0, 255, 0), 3)
                
                # Оранжевый кружок для номера
                label_radius = max(12, min(radius // 2, 25))
                cv2.circle(output, (x, y), label_radius, (0, 140, 255), -1)
                cv2.circle(output, (x, y), label_radius, (0, 100, 200), 2)
                
                # Номер круга
                label = str(i)
                font_scale = max(0.4, min(label_radius / 20, 0.8))
                thickness = max(1, int(font_scale * 2))
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = x - tw // 2
                text_y = y + th // 2
                
                cv2.putText(output, label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                           (255, 255, 255), thickness, cv2.LINE_AA)
                
                count += 1
        
        return count, detected_circles, output
    
    def detect_from_file(self, filepath):
        """Детектирование из файла"""
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Не удалось загрузить: {filepath}")
        return self.detect_circles(image)


# Тестирование
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        detector = CircleDetector()
        count, circles, result = detector.detect_from_file(sys.argv[1])
        print(f"Найдено: {count}")
        cv2.imwrite("result.jpg", result)

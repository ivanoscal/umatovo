"""
Circle Detector Module
Использует OpenCV для распознавания и подсчёта круглых объектов на изображении
"""

import cv2
import numpy as np


class CircleDetector:
    """Класс для детектирования круглых объектов на изображении"""
    
    def __init__(self, dp=1.2, min_dist=20, param1=50, param2=30, 
                 min_radius=5, max_radius=300):
        """
        Инициализация детектора кругов
        
        Args:
            dp: Обратное соотношение разрешения аккумулятора к разрешению изображения
            min_dist: Минимальное расстояние между центрами кругов
            param1: Верхний порог для детектора краёв Canny
            param2: Порог аккумулятора для центров кругов
            min_radius: Минимальный радиус круга
            max_radius: Максимальный радиус круга
        """
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius
    
    def detect_circles(self, image):
        """
        Детектирование кругов на изображении
        
        Args:
            image: Изображение в формате numpy array (BGR или RGB)
            
        Returns:
            tuple: (количество_кругов, список_кругов, обработанное_изображение)
                   каждый круг представлен как (x, y, radius)
        """
        # Создаём копию для отрисовки результатов
        output = image.copy()
        
        # Конвертируем в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Применяем размытие для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Улучшаем контраст
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Детектируем круги с помощью преобразования Хафа
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        detected_circles = []
        count = 0
        
        if circles is not None:
            # Округляем координаты до целых чисел
            circles = np.uint16(np.around(circles))
            
            for i, circle in enumerate(circles[0, :], 1):
                x, y, radius = circle
                detected_circles.append((int(x), int(y), int(radius)))
                
                # Рисуем внешний круг (зелёный, толстый)
                cv2.circle(output, (x, y), radius, (0, 255, 0), 3)
                
                # Рисуем заполненный круг для фона номера (полупрозрачный)
                overlay = output.copy()
                # Размер круга для номера зависит от радиуса найденного круга
                label_radius = max(15, min(radius // 2, 40))
                cv2.circle(overlay, (x, y), label_radius, (255, 100, 0), -1)
                cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
                
                # Рисуем номер круга
                label = str(i)
                # Размер шрифта зависит от размера круга
                font_scale = max(0.4, min(label_radius / 25, 1.5))
                thickness = max(1, int(font_scale * 2))
                
                # Получаем размер текста для центрирования
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # Позиция текста (центрируем)
                text_x = int(x - text_width / 2)
                text_y = int(y + text_height / 2)
                
                # Рисуем текст (белый)
                cv2.putText(
                    output, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA
                )
                
                count += 1
        
        return count, detected_circles, output
    
    def detect_from_file(self, filepath):
        """
        Детектирование кругов из файла изображения
        
        Args:
            filepath: Путь к файлу изображения
            
        Returns:
            tuple: (количество_кругов, список_кругов, обработанное_изображение)
        """
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {filepath}")
        
        return self.detect_circles(image)
    
    def detect_from_bytes(self, image_bytes):
        """
        Детектирование кругов из байтов изображения
        
        Args:
            image_bytes: Байты изображения
            
        Returns:
            tuple: (количество_кругов, список_кругов, обработанное_изображение)
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Не удалось декодировать изображение из байтов")
        
        return self.detect_circles(image)


def save_result(image, filepath):
    """
    Сохранение результата в файл
    
    Args:
        image: Обработанное изображение
        filepath: Путь для сохранения
    """
    cv2.imwrite(filepath, image)


# Для тестирования модуля
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python circle_detector.py <путь_к_изображению>")
        sys.exit(1)
    
    detector = CircleDetector()
    try:
        count, circles, result_image = detector.detect_from_file(sys.argv[1])
        print(f"Найдено круглых объектов: {count}")
        
        for i, (x, y, r) in enumerate(circles, 1):
            print(f"  Круг {i}: центр=({x}, {y}), радиус={r}")
        
        # Сохраняем результат
        output_path = "result_" + sys.argv[1].split("/")[-1]
        save_result(result_image, output_path)
        print(f"Результат сохранён в: {output_path}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

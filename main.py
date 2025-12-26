"""
Circle Counter - Android приложение для подсчёта круглых объектов
Использует Kivy для интерфейса и OpenCV для распознавания
"""

import os
import tempfile
from io import BytesIO

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.utils import platform

import cv2
import numpy as np

from circle_detector import CircleDetector


class CircleCounterApp(App):
    """Главное приложение для подсчёта кругов"""
    
    def build(self):
        self.title = 'Circle Counter'
        self.detector = CircleDetector()
        self.camera = None
        self.camera_active = False
        
        # Главный layout
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        # Заголовок
        title_label = Label(
            text='🔵 Circle Counter',
            font_size='28sp',
            size_hint=(1, 0.1),
            bold=True
        )
        self.layout.add_widget(title_label)
        
        # Область для отображения изображения
        self.image_widget = Image(
            size_hint=(1, 0.5),
            allow_stretch=True,
            keep_ratio=True
        )
        self.layout.add_widget(self.image_widget)
        
        # Метка с результатом
        self.result_label = Label(
            text='Выберите изображение или сделайте фото',
            font_size='20sp',
            size_hint=(1, 0.1),
            halign='center'
        )
        self.layout.add_widget(self.result_label)
        
        # Кнопки
        buttons_layout = BoxLayout(
            orientation='horizontal', 
            size_hint=(1, 0.15),
            spacing=10
        )
        
        # Кнопка камеры
        camera_btn = Button(
            text='📷 Камера',
            font_size='18sp',
            background_color=(0.2, 0.6, 1, 1)
        )
        camera_btn.bind(on_press=self.open_camera)
        buttons_layout.add_widget(camera_btn)
        
        # Кнопка галереи
        gallery_btn = Button(
            text='🖼 Галерея',
            font_size='18sp',
            background_color=(0.2, 0.8, 0.4, 1)
        )
        gallery_btn.bind(on_press=self.open_gallery)
        buttons_layout.add_widget(gallery_btn)
        
        self.layout.add_widget(buttons_layout)
        
        # Кнопка захвата (для камеры)
        self.capture_btn = Button(
            text='📸 Сделать снимок',
            font_size='18sp',
            size_hint=(1, 0.1),
            background_color=(1, 0.5, 0.2, 1),
            disabled=True
        )
        self.capture_btn.bind(on_press=self.capture_photo)
        self.layout.add_widget(self.capture_btn)
        
        return self.layout
    
    def open_camera(self, instance):
        """Открытие камеры для съёмки"""
        if platform == 'android':
            self._open_android_camera()
        else:
            self._open_desktop_camera()
    
    def _open_android_camera(self):
        """Открытие камеры на Android"""
        try:
            from android.permissions import request_permissions, Permission
            from plyer import camera
            
            def on_permissions(permissions, grants):
                if all(grants):
                    # Создаём временный файл для фото
                    temp_dir = tempfile.gettempdir()
                    self.temp_photo_path = os.path.join(temp_dir, 'circle_photo.jpg')
                    camera.take_picture(
                        filename=self.temp_photo_path,
                        on_complete=self._on_camera_complete
                    )
            
            request_permissions([
                Permission.CAMERA,
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE
            ], on_permissions)
            
        except ImportError:
            self.result_label.text = 'Камера недоступна на этом устройстве'
    
    def _open_desktop_camera(self):
        """Открытие камеры на десктопе (для тестирования)"""
        if self.camera_active:
            self._stop_camera()
            return
        
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.result_label.text = 'Не удалось открыть камеру'
            return
        
        self.camera_active = True
        self.capture_btn.disabled = False
        self.result_label.text = 'Камера активна. Нажмите "Сделать снимок"'
        
        # Запускаем обновление кадров
        Clock.schedule_interval(self._update_camera_frame, 1.0 / 30.0)
    
    def _update_camera_frame(self, dt):
        """Обновление кадра с камеры"""
        if not self.camera_active or self.camera is None:
            return False
        
        ret, frame = self.camera.read()
        if ret:
            # Конвертируем BGR в RGB для отображения
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Переворачиваем по вертикали для Kivy
            frame_rgb = cv2.flip(frame_rgb, 0)
            
            # Создаём текстуру
            texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]),
                colorfmt='rgb'
            )
            texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            self.image_widget.texture = texture
    
    def _stop_camera(self):
        """Остановка камеры"""
        self.camera_active = False
        self.capture_btn.disabled = True
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        Clock.unschedule(self._update_camera_frame)
    
    def capture_photo(self, instance):
        """Захват фото с камеры"""
        if self.camera is None or not self.camera_active:
            return
        
        ret, frame = self.camera.read()
        if ret:
            self._stop_camera()
            self._process_image(frame)
    
    def _on_camera_complete(self, filepath):
        """Callback после съёмки на Android"""
        if filepath and os.path.exists(filepath):
            image = cv2.imread(filepath)
            if image is not None:
                self._process_image(image)
            else:
                self.result_label.text = 'Ошибка загрузки фото'
    
    def open_gallery(self, instance):
        """Открытие галереи для выбора изображения"""
        if platform == 'android':
            self._open_android_gallery()
        else:
            self._open_desktop_gallery()
    
    def _open_android_gallery(self):
        """Открытие галереи на Android"""
        try:
            from android.permissions import request_permissions, Permission
            from plyer import filechooser
            
            def on_permissions(permissions, grants):
                if all(grants):
                    filechooser.open_file(
                        on_selection=self._on_file_selected,
                        filters=['*.png', '*.jpg', '*.jpeg']
                    )
            
            request_permissions([
                Permission.READ_EXTERNAL_STORAGE
            ], on_permissions)
            
        except ImportError:
            self._open_desktop_gallery()
    
    def _open_desktop_gallery(self):
        """Открытие файлового диалога на десктопе"""
        self._stop_camera()
        
        content = BoxLayout(orientation='vertical')
        
        # Определяем стартовую директорию
        if platform == 'android':
            start_path = '/sdcard/DCIM'
        else:
            start_path = os.path.expanduser('~')
        
        filechooser = FileChooserListView(
            path=start_path,
            filters=['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        )
        content.add_widget(filechooser)
        
        # Кнопки
        btn_layout = BoxLayout(size_hint=(1, 0.1), spacing=10)
        
        select_btn = Button(text='Выбрать')
        cancel_btn = Button(text='Отмена')
        
        btn_layout.add_widget(select_btn)
        btn_layout.add_widget(cancel_btn)
        content.add_widget(btn_layout)
        
        popup = Popup(
            title='Выберите изображение',
            content=content,
            size_hint=(0.9, 0.9)
        )
        
        def on_select(instance):
            if filechooser.selection:
                popup.dismiss()
                self._on_file_selected(filechooser.selection)
        
        def on_cancel(instance):
            popup.dismiss()
        
        select_btn.bind(on_press=on_select)
        cancel_btn.bind(on_press=on_cancel)
        
        popup.open()
    
    def _on_file_selected(self, selection):
        """Обработка выбранного файла"""
        if selection and len(selection) > 0:
            filepath = selection[0]
            if os.path.exists(filepath):
                image = cv2.imread(filepath)
                if image is not None:
                    self._process_image(image)
                else:
                    self.result_label.text = 'Ошибка загрузки изображения'
            else:
                self.result_label.text = 'Файл не найден'
    
    def _process_image(self, image):
        """Обработка изображения и подсчёт кругов"""
        try:
            # Детектируем круги
            count, circles, result_image = self.detector.detect_circles(image)
            
            # Обновляем результат
            if count == 0:
                self.result_label.text = 'Круглых объектов не найдено'
            elif count == 1:
                self.result_label.text = f'Найден 1 круглый объект'
            else:
                self.result_label.text = f'Найдено {count} круглых объектов'
            
            # Отображаем результат
            self._display_image(result_image)
            
        except Exception as e:
            self.result_label.text = f'Ошибка обработки: {str(e)}'
    
    def _display_image(self, image):
        """Отображение изображения в виджете"""
        # Конвертируем BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Переворачиваем для Kivy
        image_rgb = cv2.flip(image_rgb, 0)
        
        # Создаём текстуру
        texture = Texture.create(
            size=(image.shape[1], image.shape[0]),
            colorfmt='rgb'
        )
        texture.blit_buffer(image_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.image_widget.texture = texture
    
    def on_stop(self):
        """Очистка при закрытии приложения"""
        self._stop_camera()


if __name__ == '__main__':
    CircleCounterApp().run()


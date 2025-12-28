"""
Circle Counter - Android приложение для подсчёта круглых объектов
Использует Kivy для интерфейса и OpenCV для распознавания
"""

import os
import tempfile

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.utils import platform
from kivy.logger import Logger

import cv2
import numpy as np

from circle_detector import CircleDetector


class CircleCounterApp(App):
    """Главное приложение для подсчёта кругов"""
    
    def build(self):
        self.title = 'Circle Counter'
        self.detector = CircleDetector()
        self.temp_photo_path = None
        
        # Главный layout
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        # Заголовок
        title_label = Label(
            text='Circle Counter',
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
            font_size='18sp',
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
            text='Камера',
            font_size='18sp',
            background_color=(0.2, 0.6, 1, 1)
        )
        camera_btn.bind(on_press=self.open_camera)
        buttons_layout.add_widget(camera_btn)
        
        # Кнопка галереи
        gallery_btn = Button(
            text='Галерея',
            font_size='18sp',
            background_color=(0.2, 0.8, 0.4, 1)
        )
        gallery_btn.bind(on_press=self.open_gallery)
        buttons_layout.add_widget(gallery_btn)
        
        self.layout.add_widget(buttons_layout)
        
        # Запрашиваем разрешения на Android при старте
        if platform == 'android':
            Clock.schedule_once(self._request_android_permissions, 1)
        
        return self.layout
    
    def _request_android_permissions(self, dt):
        """Запрос разрешений на Android"""
        if platform == 'android':
            try:
                from android.permissions import request_permissions, Permission
                request_permissions([
                    Permission.CAMERA,
                    Permission.READ_EXTERNAL_STORAGE,
                    Permission.WRITE_EXTERNAL_STORAGE,
                ])
                Logger.info("Permissions requested")
            except Exception as e:
                Logger.error(f"Permission request error: {e}")
    
    def open_camera(self, instance):
        """Открытие камеры"""
        Logger.info("Camera button pressed")
        
        if platform == 'android':
            self._android_camera()
        else:
            self._desktop_camera()
    
    def _android_camera(self):
        """Камера на Android через plyer"""
        try:
            from plyer import camera
            
            # Создаём путь для фото
            if platform == 'android':
                from android.storage import app_storage_path
                temp_dir = app_storage_path()
            else:
                temp_dir = tempfile.gettempdir()
            
            self.temp_photo_path = os.path.join(temp_dir, 'circle_photo.jpg')
            Logger.info(f"Photo path: {self.temp_photo_path}")
            
            # Делаем фото
            camera.take_picture(
                filename=self.temp_photo_path,
                on_complete=self._on_camera_complete
            )
            
        except Exception as e:
            Logger.error(f"Camera error: {e}")
            self.result_label.text = f'Ошибка камеры: {str(e)}'
    
    def _desktop_camera(self):
        """Камера на десктопе (для тестирования)"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.result_label.text = 'Камера недоступна'
                return
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                self._process_image(frame)
            else:
                self.result_label.text = 'Не удалось сделать снимок'
                
        except Exception as e:
            self.result_label.text = f'Ошибка: {str(e)}'
    
    def _on_camera_complete(self, filepath):
        """Callback после съёмки фото"""
        Logger.info(f"Camera complete: {filepath}")
        
        if filepath and os.path.exists(filepath):
            try:
                image = cv2.imread(filepath)
                if image is not None:
                    self._process_image(image)
                else:
                    self.result_label.text = 'Не удалось загрузить фото'
            except Exception as e:
                Logger.error(f"Load error: {e}")
                self.result_label.text = f'Ошибка: {str(e)}'
        else:
            self.result_label.text = 'Фото не сохранено'
    
    def open_gallery(self, instance):
        """Открытие галереи"""
        Logger.info("Gallery button pressed")
        
        if platform == 'android':
            self._android_gallery()
        else:
            self._desktop_gallery()
    
    def _android_gallery(self):
        """Галерея на Android через plyer"""
        try:
            from plyer import filechooser
            
            filechooser.open_file(
                on_selection=self._on_file_selected,
                filters=[("Images", "*.png", "*.jpg", "*.jpeg")],
                mime_type="image/*"
            )
            
        except Exception as e:
            Logger.error(f"Gallery error: {e}")
            self.result_label.text = f'Ошибка галереи: {str(e)}'
    
    def _desktop_gallery(self):
        """Файловый диалог на десктопе"""
        content = BoxLayout(orientation='vertical')
        
        start_path = os.path.expanduser('~')
        
        filechooser = FileChooserListView(
            path=start_path,
            filters=['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        )
        content.add_widget(filechooser)
        
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
        
        def on_select(inst):
            if filechooser.selection:
                popup.dismiss()
                self._on_file_selected(filechooser.selection)
        
        def on_cancel(inst):
            popup.dismiss()
        
        select_btn.bind(on_press=on_select)
        cancel_btn.bind(on_press=on_cancel)
        
        popup.open()
    
    def _on_file_selected(self, selection):
        """Обработка выбранного файла"""
        Logger.info(f"File selected: {selection}")
        
        if selection and len(selection) > 0:
            filepath = selection[0]
            
            if os.path.exists(filepath):
                try:
                    image = cv2.imread(filepath)
                    if image is not None:
                        self._process_image(image)
                    else:
                        self.result_label.text = 'Не удалось загрузить изображение'
                except Exception as e:
                    Logger.error(f"Load error: {e}")
                    self.result_label.text = f'Ошибка: {str(e)}'
            else:
                self.result_label.text = 'Файл не найден'
    
    def _process_image(self, image):
        """Обработка изображения и подсчёт кругов"""
        Logger.info("Processing image")
        
        try:
            # Детектируем круги
            count, circles, result_image = self.detector.detect_circles(image)
            
            # Обновляем результат
            if count == 0:
                self.result_label.text = 'Круглых объектов не найдено'
            elif count == 1:
                self.result_label.text = 'Найден 1 круглый объект'
            else:
                self.result_label.text = f'Найдено {count} круглых объектов'
            
            # Отображаем результат
            self._display_image(result_image)
            
        except Exception as e:
            Logger.error(f"Process error: {e}")
            self.result_label.text = f'Ошибка обработки: {str(e)}'
    
    def _display_image(self, image):
        """Отображение изображения"""
        try:
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
            
        except Exception as e:
            Logger.error(f"Display error: {e}")


if __name__ == '__main__':
    CircleCounterApp().run()

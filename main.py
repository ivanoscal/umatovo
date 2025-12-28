"""
Circle Counter - Android приложение для подсчёта круглых объектов
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
from kivy.core.window import Window

import cv2
import numpy as np

from circle_detector import CircleDetector


class CircleCounterApp(App):
    """Главное приложение"""
    
    def build(self):
        self.title = 'Circle Counter'
        self.detector = CircleDetector()
        self.current_image = None
        
        # Устанавливаем цвет фона
        Window.clearcolor = (0.1, 0.1, 0.1, 1)
        
        # Главный layout
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Заголовок
        self.layout.add_widget(Label(
            text='Circle Counter',
            font_size='24sp',
            size_hint=(1, 0.08),
            bold=True
        ))
        
        # Область для изображения
        self.image_widget = Image(
            size_hint=(1, 0.6),
            allow_stretch=True,
            keep_ratio=True
        )
        self.layout.add_widget(self.image_widget)
        
        # Результат
        self.result_label = Label(
            text='Выберите фото для анализа',
            font_size='18sp',
            size_hint=(1, 0.1),
            halign='center',
            valign='middle'
        )
        self.result_label.bind(size=self.result_label.setter('text_size'))
        self.layout.add_widget(self.result_label)
        
        # Кнопки
        btn_layout = BoxLayout(size_hint=(1, 0.15), spacing=10)
        
        camera_btn = Button(
            text='Камера',
            font_size='18sp',
            background_color=(0.2, 0.5, 0.8, 1),
            background_normal=''
        )
        camera_btn.bind(on_press=self.open_camera)
        btn_layout.add_widget(camera_btn)
        
        gallery_btn = Button(
            text='Галерея',
            font_size='18sp',
            background_color=(0.2, 0.6, 0.3, 1),
            background_normal=''
        )
        gallery_btn.bind(on_press=self.open_gallery)
        btn_layout.add_widget(gallery_btn)
        
        self.layout.add_widget(btn_layout)
        
        # Запрос разрешений на Android
        if platform == 'android':
            Clock.schedule_once(self._request_permissions, 0.5)
        
        return self.layout
    
    def _request_permissions(self, dt):
        """Запрос разрешений Android"""
        try:
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.CAMERA,
                Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE,
            ])
        except Exception as e:
            Logger.error(f"Permissions: {e}")
    
    def open_camera(self, instance):
        """Открыть камеру"""
        if platform == 'android':
            try:
                from plyer import camera
                
                # Путь для сохранения фото
                try:
                    from android.storage import app_storage_path
                    photo_dir = app_storage_path()
                except:
                    photo_dir = tempfile.gettempdir()
                
                photo_path = os.path.join(photo_dir, 'photo.jpg')
                
                camera.take_picture(
                    filename=photo_path,
                    on_complete=self._on_photo_taken
                )
            except Exception as e:
                Logger.error(f"Camera: {e}")
                self.result_label.text = f'Ошибка камеры: {e}'
        else:
            # Десктоп - снимок с веб-камеры
            try:
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self._process_and_display(frame)
                else:
                    self.result_label.text = 'Камера недоступна'
            except Exception as e:
                self.result_label.text = f'Ошибка: {e}'
    
    def _on_photo_taken(self, filepath):
        """Фото сделано"""
        Logger.info(f"Photo: {filepath}")
        if filepath and os.path.exists(filepath):
            self._load_and_process(filepath)
        else:
            self.result_label.text = 'Фото не сохранено'
    
    def open_gallery(self, instance):
        """Открыть галерею"""
        if platform == 'android':
            try:
                from plyer import filechooser
                filechooser.open_file(
                    on_selection=self._on_file_selected,
                    mime_type="image/*"
                )
            except Exception as e:
                Logger.error(f"Gallery: {e}")
                self.result_label.text = f'Ошибка: {e}'
        else:
            self._show_file_chooser()
    
    def _show_file_chooser(self):
        """Файловый диалог для десктопа"""
        content = BoxLayout(orientation='vertical')
        
        fc = FileChooserListView(
            path=os.path.expanduser('~'),
            filters=['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        )
        content.add_widget(fc)
        
        btns = BoxLayout(size_hint=(1, 0.1), spacing=5)
        
        select_btn = Button(text='Выбрать')
        cancel_btn = Button(text='Отмена')
        btns.add_widget(select_btn)
        btns.add_widget(cancel_btn)
        content.add_widget(btns)
        
        popup = Popup(title='Выберите изображение', content=content, size_hint=(0.95, 0.95))
        
        def on_select(inst):
            if fc.selection:
                popup.dismiss()
                self._on_file_selected(fc.selection)
        
        select_btn.bind(on_press=on_select)
        cancel_btn.bind(on_press=lambda x: popup.dismiss())
        popup.open()
    
    def _on_file_selected(self, selection):
        """Файл выбран"""
        Logger.info(f"Selected: {selection}")
        if selection:
            filepath = selection[0] if isinstance(selection, list) else selection
            if os.path.exists(filepath):
                self._load_and_process(filepath)
    
    def _load_and_process(self, filepath):
        """Загрузить и обработать изображение"""
        try:
            image = cv2.imread(filepath)
            if image is not None:
                self._process_and_display(image)
            else:
                self.result_label.text = 'Не удалось загрузить изображение'
        except Exception as e:
            Logger.error(f"Load: {e}")
            self.result_label.text = f'Ошибка загрузки: {e}'
    
    def _process_and_display(self, image):
        """Обработать и отобразить"""
        try:
            self.result_label.text = 'Обработка...'
            
            # Детектируем круги
            count, circles, result_image = self.detector.detect_circles(image)
            
            # Обновляем текст
            if count == 0:
                self.result_label.text = 'Круглых объектов не найдено'
            elif count == 1:
                self.result_label.text = 'Найден 1 круглый объект'
            elif count < 5:
                self.result_label.text = f'Найдено {count} круглых объекта'
            else:
                self.result_label.text = f'Найдено {count} круглых объектов'
            
            # Отображаем результат
            self._show_image(result_image)
            
        except Exception as e:
            Logger.error(f"Process: {e}")
            self.result_label.text = f'Ошибка: {e}'
    
    def _show_image(self, cv_image):
        """Отобразить OpenCV изображение в Kivy"""
        try:
            # Получаем размеры
            height, width = cv_image.shape[:2]
            
            # BGR -> RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Переворачиваем вертикально для Kivy (Kivy использует нижний левый угол как начало)
            flipped = cv2.flip(rgb_image, 0)
            
            # Преобразуем в bytes
            buf = flipped.tobytes()
            
            # Создаём текстуру
            texture = Texture.create(size=(width, height), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            
            # Применяем к виджету
            self.image_widget.texture = texture
            
            Logger.info(f"Image displayed: {width}x{height}")
            
        except Exception as e:
            Logger.error(f"Display: {e}")
            self.result_label.text = f'Ошибка отображения: {e}'


if __name__ == '__main__':
    CircleCounterApp().run()

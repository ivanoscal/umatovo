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


class TouchableImage(Image):
    """Изображение с поддержкой тапов для добавления кругов"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = None
    
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and self.app and self.app.current_image is not None:
            # Преобразуем координаты тапа в координаты изображения
            self.app.add_manual_circle(touch.pos)
            return True
        return super().on_touch_down(touch)


class CircleCounterApp(App):
    """Главное приложение"""
    
    def build(self):
        self.title = 'Circle Counter'
        self.detector = CircleDetector()
        self.current_image = None
        self.result_image = None
        self.detected_count = 0
        self.manual_count = 0
        self.circles = []
        self.median_radius = 15
        
        Window.clearcolor = (0.1, 0.1, 0.1, 1)
        
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Заголовок
        self.layout.add_widget(Label(
            text='Circle Counter',
            font_size='24sp',
            size_hint=(1, 0.06),
            bold=True
        ))
        
        # Изображение с поддержкой тапов
        self.image_widget = TouchableImage(
            size_hint=(1, 0.6),
            allow_stretch=True,
            keep_ratio=True
        )
        self.image_widget.app = self
        self.layout.add_widget(self.image_widget)
        
        # Подсказка
        self.hint_label = Label(
            text='Тапните на пропущенный объект чтобы добавить',
            font_size='12sp',
            size_hint=(1, 0.04),
            color=(0.7, 0.7, 0.7, 1)
        )
        self.layout.add_widget(self.hint_label)
        
        # Результат
        self.result_label = Label(
            text='Выберите фото для анализа',
            font_size='18sp',
            size_hint=(1, 0.08),
            halign='center',
            valign='middle'
        )
        self.result_label.bind(size=self.result_label.setter('text_size'))
        self.layout.add_widget(self.result_label)
        
        # Кнопки
        btn_layout = BoxLayout(size_hint=(1, 0.12), spacing=10)
        
        camera_btn = Button(
            text='Камера',
            font_size='16sp',
            background_color=(0.2, 0.5, 0.8, 1),
            background_normal=''
        )
        camera_btn.bind(on_press=self.open_camera)
        btn_layout.add_widget(camera_btn)
        
        gallery_btn = Button(
            text='Галерея',
            font_size='16sp',
            background_color=(0.2, 0.6, 0.3, 1),
            background_normal=''
        )
        gallery_btn.bind(on_press=self.open_gallery)
        btn_layout.add_widget(gallery_btn)
        
        reset_btn = Button(
            text='Сброс',
            font_size='16sp',
            background_color=(0.6, 0.3, 0.3, 1),
            background_normal=''
        )
        reset_btn.bind(on_press=self.reset_manual)
        btn_layout.add_widget(reset_btn)
        
        self.layout.add_widget(btn_layout)
        
        if platform == 'android':
            Clock.schedule_once(self._request_permissions, 0.5)
        
        return self.layout
    
    def _request_permissions(self, dt):
        try:
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.CAMERA,
                Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE,
            ])
        except Exception as e:
            Logger.error(f"Permissions: {e}")
    
    def add_manual_circle(self, touch_pos):
        """Добавление круга по тапу"""
        if self.result_image is None:
            return
        
        # Получаем размеры виджета и изображения
        widget_w, widget_h = self.image_widget.size
        widget_x, widget_y = self.image_widget.pos
        
        img_h, img_w = self.result_image.shape[:2]
        
        # Вычисляем масштаб и смещение (keep_ratio=True)
        scale_w = widget_w / img_w
        scale_h = widget_h / img_h
        scale = min(scale_w, scale_h)
        
        displayed_w = img_w * scale
        displayed_h = img_h * scale
        
        offset_x = widget_x + (widget_w - displayed_w) / 2
        offset_y = widget_y + (widget_h - displayed_h) / 2
        
        # Преобразуем координаты тапа в координаты изображения
        touch_x, touch_y = touch_pos
        
        img_x = (touch_x - offset_x) / scale
        img_y = img_h - (touch_y - offset_y) / scale  # Инвертируем Y
        
        # Проверяем что тап внутри изображения
        if 0 <= img_x < img_w and 0 <= img_y < img_h:
            self.manual_count += 1
            total = self.detected_count + self.manual_count
            
            # Добавляем круг на изображение
            x, y = int(img_x), int(img_y)
            r = int(self.median_radius)
            
            # Рисуем круг (синий для ручных)
            cv2.circle(self.result_image, (x, y), r, (255, 100, 0), 2)
            cv2.circle(self.result_image, (x, y), 9, (255, 100, 0), -1)
            
            label = str(total)
            font_scale = 0.32
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.putText(self.result_image, label, (x - tw//2, y + th//2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            
            self._update_display()
            self._update_result_text()
            
            Logger.info(f"Manual circle added at ({x}, {y}), total: {total}")
    
    def reset_manual(self, instance):
        """Сброс ручных добавлений"""
        if self.current_image is not None:
            self.manual_count = 0
            self._process_and_display(self.current_image.copy())
    
    def _update_result_text(self):
        """Обновление текста результата"""
        total = self.detected_count + self.manual_count
        
        if total == 0:
            self.result_label.text = 'Круглых объектов не найдено'
        elif self.manual_count > 0:
            self.result_label.text = f'Найдено: {self.detected_count} + {self.manual_count} = {total}'
        else:
            if total == 1:
                self.result_label.text = 'Найден 1 круглый объект'
            elif total < 5:
                self.result_label.text = f'Найдено {total} круглых объекта'
            else:
                self.result_label.text = f'Найдено {total} круглых объектов'
    
    def open_camera(self, instance):
        if platform == 'android':
            try:
                from plyer import camera
                try:
                    from android.storage import app_storage_path
                    photo_dir = app_storage_path()
                except:
                    photo_dir = tempfile.gettempdir()
                
                photo_path = os.path.join(photo_dir, 'photo.jpg')
                camera.take_picture(filename=photo_path, on_complete=self._on_photo_taken)
            except Exception as e:
                Logger.error(f"Camera: {e}")
                self.result_label.text = f'Ошибка камеры: {e}'
        else:
            try:
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.current_image = frame.copy()
                    self._process_and_display(frame)
            except Exception as e:
                self.result_label.text = f'Ошибка: {e}'
    
    def _on_photo_taken(self, filepath):
        if filepath and os.path.exists(filepath):
            self._load_and_process(filepath)
    
    def open_gallery(self, instance):
        if platform == 'android':
            try:
                from plyer import filechooser
                filechooser.open_file(on_selection=self._on_file_selected, mime_type="image/*")
            except Exception as e:
                Logger.error(f"Gallery: {e}")
                self.result_label.text = f'Ошибка: {e}'
        else:
            self._show_file_chooser()
    
    def _show_file_chooser(self):
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
        if selection:
            filepath = selection[0] if isinstance(selection, list) else selection
            if os.path.exists(filepath):
                self._load_and_process(filepath)
    
    def _load_and_process(self, filepath):
        try:
            image = cv2.imread(filepath)
            if image is not None:
                self.current_image = image.copy()
                self._process_and_display(image)
        except Exception as e:
            Logger.error(f"Load: {e}")
            self.result_label.text = f'Ошибка: {e}'
    
    def _process_and_display(self, image):
        try:
            self.result_label.text = 'Обработка...'
            self.manual_count = 0
            
            count, circles, result_image = self.detector.detect_circles(image)
            
            self.detected_count = count
            self.circles = circles
            self.result_image = result_image
            
            # Вычисляем медианный радиус для ручного добавления
            if circles:
                self.median_radius = np.median([c[2] for c in circles])
            
            self._update_result_text()
            self._update_display()
            
        except Exception as e:
            Logger.error(f"Process: {e}")
            self.result_label.text = f'Ошибка: {e}'
    
    def _update_display(self):
        """Обновление отображения изображения"""
        if self.result_image is None:
            return
        
        try:
            height, width = self.result_image.shape[:2]
            rgb_image = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
            flipped = cv2.flip(rgb_image, 0)
            buf = flipped.tobytes()
            
            texture = Texture.create(size=(width, height), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.image_widget.texture = texture
            
        except Exception as e:
            Logger.error(f"Display: {e}")


if __name__ == '__main__':
    CircleCounterApp().run()

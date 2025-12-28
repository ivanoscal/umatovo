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
    """Изображение с поддержкой тапов"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = None
    
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and self.app and self.app.current_image is not None:
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
        self.base_result_image = None  # Изображение до ручных добавлений
        self.detected_count = 0
        self.manual_circles = []  # Список ручных кругов для отмены
        self.circles = []
        self.median_radius = 15
        
        Window.clearcolor = (0.1, 0.1, 0.1, 1)
        
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=8)
        
        # Заголовок
        self.layout.add_widget(Label(
            text='Circle Counter',
            font_size='24sp',
            size_hint=(1, 0.05),
            bold=True
        ))
        
        # Изображение
        self.image_widget = TouchableImage(
            size_hint=(1, 0.62),
            allow_stretch=True,
            keep_ratio=True
        )
        self.image_widget.app = self
        self.layout.add_widget(self.image_widget)
        
        # Подсказка
        self.hint_label = Label(
            text='Тапните чтобы добавить пропущенный объект',
            font_size='11sp',
            size_hint=(1, 0.03),
            color=(0.6, 0.6, 0.6, 1)
        )
        self.layout.add_widget(self.hint_label)
        
        # Результат
        self.result_label = Label(
            text='Выберите фото для анализа',
            font_size='18sp',
            size_hint=(1, 0.07),
            halign='center'
        )
        self.layout.add_widget(self.result_label)
        
        # Кнопки ряд 1: Камера и Галерея
        btn_layout1 = BoxLayout(size_hint=(1, 0.10), spacing=8)
        
        camera_btn = Button(
            text='Камера',
            font_size='15sp',
            background_color=(0.2, 0.5, 0.8, 1),
            background_normal=''
        )
        camera_btn.bind(on_press=self.open_camera)
        btn_layout1.add_widget(camera_btn)
        
        gallery_btn = Button(
            text='Галерея',
            font_size='15sp',
            background_color=(0.2, 0.6, 0.3, 1),
            background_normal=''
        )
        gallery_btn.bind(on_press=self.open_gallery)
        btn_layout1.add_widget(gallery_btn)
        
        self.layout.add_widget(btn_layout1)
        
        # Кнопки ряд 2: Назад и Сброс
        btn_layout2 = BoxLayout(size_hint=(1, 0.10), spacing=8)
        
        undo_btn = Button(
            text='← Назад',
            font_size='15sp',
            background_color=(0.5, 0.4, 0.2, 1),
            background_normal=''
        )
        undo_btn.bind(on_press=self.undo_last)
        btn_layout2.add_widget(undo_btn)
        
        reset_btn = Button(
            text='Сброс всех',
            font_size='15sp',
            background_color=(0.6, 0.25, 0.25, 1),
            background_normal=''
        )
        reset_btn.bind(on_press=self.reset_manual)
        btn_layout2.add_widget(reset_btn)
        
        self.layout.add_widget(btn_layout2)
        
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
        
        widget_w, widget_h = self.image_widget.size
        widget_x, widget_y = self.image_widget.pos
        img_h, img_w = self.result_image.shape[:2]
        
        scale_w = widget_w / img_w
        scale_h = widget_h / img_h
        scale = min(scale_w, scale_h)
        
        displayed_w = img_w * scale
        displayed_h = img_h * scale
        
        offset_x = widget_x + (widget_w - displayed_w) / 2
        offset_y = widget_y + (widget_h - displayed_h) / 2
        
        touch_x, touch_y = touch_pos
        img_x = (touch_x - offset_x) / scale
        img_y = img_h - (touch_y - offset_y) / scale
        
        if 0 <= img_x < img_w and 0 <= img_y < img_h:
            x, y = int(img_x), int(img_y)
            r = int(self.median_radius)
            
            # Сохраняем для отмены
            self.manual_circles.append((x, y, r))
            
            total = self.detected_count + len(self.manual_circles)
            
            # Рисуем белый круг
            cv2.circle(self.result_image, (x, y), r, (255, 255, 255), 2)
            cv2.circle(self.result_image, (x, y), 9, (255, 255, 255), -1)
            
            label = str(total)
            font_scale = 0.32
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.putText(self.result_image, label, (x - tw//2, y + th//2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
            
            self._update_display()
            self._update_result_text()
    
    def undo_last(self, instance):
        """Отмена последнего добавленного круга"""
        if not self.manual_circles or self.base_result_image is None:
            return
        
        # Удаляем последний круг
        self.manual_circles.pop()
        
        # Перерисовываем с нуля
        self.result_image = self.base_result_image.copy()
        
        # Добавляем оставшиеся ручные круги
        for i, (x, y, r) in enumerate(self.manual_circles):
            num = self.detected_count + i + 1
            cv2.circle(self.result_image, (x, y), r, (255, 255, 255), 2)
            cv2.circle(self.result_image, (x, y), 9, (255, 255, 255), -1)
            label = str(num)
            font_scale = 0.32
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.putText(self.result_image, label, (x - tw//2, y + th//2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        
        self._update_display()
        self._update_result_text()
    
    def reset_manual(self, instance):
        """Сброс всех ручных добавлений"""
        if self.base_result_image is not None:
            self.manual_circles = []
            self.result_image = self.base_result_image.copy()
            self._update_display()
            self._update_result_text()
    
    def _update_result_text(self):
        total = self.detected_count + len(self.manual_circles)
        manual = len(self.manual_circles)
        
        if total == 0:
            self.result_label.text = 'Круглых объектов не найдено'
        elif manual > 0:
            self.result_label.text = f'Найдено: {self.detected_count} + {manual} = {total}'
        else:
            self.result_label.text = f'Найдено: {total}'
    
    def open_camera(self, instance):
        """Открытие камеры"""
        if platform == 'android':
            try:
                # Используем Intent напрямую для камеры
                from android import activity
                from jnius import autoclass
                
                Intent = autoclass('android.content.Intent')
                MediaStore = autoclass('android.provider.MediaStore')
                
                intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                activity.start_activity_for_result(intent, 1)
                
                # Устанавливаем обработчик результата
                activity.bind(on_activity_result=self._on_camera_result)
                
            except Exception as e:
                Logger.error(f"Camera Intent: {e}")
                # Fallback - используем галерею
                self.result_label.text = 'Камера недоступна. Используйте галерею.'
        else:
            # Десктоп
            try:
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.current_image = frame.copy()
                    self._process_and_display(frame)
                else:
                    self.result_label.text = 'Камера недоступна'
            except Exception as e:
                self.result_label.text = f'Ошибка: {e}'
    
    def _on_camera_result(self, request_code, result_code, intent):
        """Обработка результата камеры"""
        if request_code == 1 and intent:
            try:
                from jnius import autoclass
                
                # Получаем bitmap из intent
                extras = intent.getExtras()
                if extras:
                    bitmap = extras.get('data')
                    if bitmap:
                        # Конвертируем Bitmap в numpy array
                        width = bitmap.getWidth()
                        height = bitmap.getHeight()
                        
                        Bitmap = autoclass('android.graphics.Bitmap')
                        ByteArrayOutputStream = autoclass('java.io.ByteArrayOutputStream')
                        CompressFormat = autoclass('android.graphics.Bitmap$CompressFormat')
                        
                        stream = ByteArrayOutputStream()
                        bitmap.compress(CompressFormat.PNG, 100, stream)
                        byte_array = stream.toByteArray()
                        
                        # Декодируем в OpenCV
                        nparr = np.frombuffer(bytes(byte_array), np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if image is not None:
                            self.current_image = image.copy()
                            self._process_and_display(image)
                            
            except Exception as e:
                Logger.error(f"Camera result: {e}")
                self.result_label.text = f'Ошибка: {e}'
    
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
            self.manual_circles = []
            
            count, circles, result_image = self.detector.detect_circles(image)
            
            self.detected_count = count
            self.circles = circles
            self.result_image = result_image
            self.base_result_image = result_image.copy()  # Сохраняем базовое
            
            if circles:
                self.median_radius = int(np.median([c[2] for c in circles]))
            
            self._update_result_text()
            self._update_display()
            
        except Exception as e:
            Logger.error(f"Process: {e}")
            self.result_label.text = f'Ошибка: {e}'
    
    def _update_display(self):
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

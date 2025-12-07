import random
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QTimer, QPointF

class FloatingPixelsWidget(QWidget):
    def __init__(self, parent=None, pixel_count=100):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.pixels = []
        self.pixel_count = pixel_count
        self.init_pixels()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(30)  # ~33 FPS

    def init_pixels(self):
        self.pixels = []
        w = self.width() or 800
        h = self.height() or 480
        for _ in range(self.pixel_count):
            pos = QPointF(random.uniform(0, w), random.uniform(0, h))
            vel = QPointF(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))
            size = random.uniform(1, 3)
            alpha = random.uniform(0.3, 1.0)
            self.pixels.append({'pos': pos, 'vel': vel, 'size': size, 'alpha': alpha})

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self.init_pixels()

    def animate(self):
        w = self.width()
        h = self.height()
        for p in self.pixels:
            p['pos'] += p['vel']
            if p['pos'].x() < 0 or p['pos'].x() > w:
                p['vel'].setX(-p['vel'].x())
            if p['pos'].y() < 0 or p['pos'].y() > h:
                p['vel'].setY(-p['vel'].y())
        self.update()

    def paintEvent(self, event):  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Qt.GlobalColor.white)
        for p in self.pixels:
            color = QColor(255, 255, 255)
            color.setAlphaF(p['alpha'])
            painter.setBrush(color)
            size = p['size']
            pos = p['pos']
            painter.drawEllipse(pos, size, size)
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget


class GalleryImage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.is_deleted = False
        self.image_path = None
        self.image_size = 250
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(self.image_size, self.image_size)
        self.checkbox = QCheckBox("")
        self.checkbox.setCheckable(True)
        self.checkbox.setFixedSize(20, 20)
        self.checkbox.toggled.connect(self.toggle_border)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.checkbox, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.image_label)
        self.setFixedSize(self.image_size + 20, self.image_size + 20)
        self.setStyleSheet("background-color: white; border: 2px solid white; border-radius: 10px;")

    def toggle_border(self, checked):
        if checked:
            self.setStyleSheet("background-color: white; border: 3px solid red; border-radius: 10px;")
        else:
            self.setStyleSheet("background-color: white; border: 2px solid white; border-radius: 10px;")

    def setImagePath(self, image_path):
        self.image_path = image_path

    def setPixmap(self, pixmap):
        if pixmap is None:
            self.image_label.clear()
            self.hideAll()
        else:
            self.image_label.setPixmap(pixmap)
            self.showAll()

    def isChecked(self):
        return self.checkbox.isChecked()

    def check(self):
        self.checkbox.setChecked(True)

    def unCheck(self):
        self.checkbox.setChecked(False)

    def hideAll(self):
        self.checkbox.hide()
        self.image_label.hide()

    def showAll(self):
        self.checkbox.show()
        self.image_label.show()

    def init(self):
        self.unCheck()
        self.showAll()
        self.is_deleted = False

    def delete(self):
        self.setPixmap(None)
        self.is_deleted = True

    def isDeleted(self):
        return self.is_deleted


class GalleryImageBufferMgr:
    def __init__(self, buf_size=100):
        self.buffer: list[GalleryImage] = []
        self.curr_index = 0
        for _ in range(buf_size):
            self.buffer.append(GalleryImage())

    def get(self):
        if self.curr_index >= len(self.buffer):
            for _ in range(25):
                self.buffer.append(GalleryImage())
        ret = self.curr_index
        self.curr_index += 1
        self.buffer[ret].init()
        return self.buffer[ret]

    def reset(self):
        for i in range(self.curr_index):
            self.buffer[i].delete()
        self.curr_index = 0

    def inUseWidgets(self):
        widgets = []
        for i in range(self.curr_index):
            if not self.buffer[i].isDeleted():
                widgets.append(self.buffer[i])
        return widgets

    def selectAllInUseWidgets(self):
        for widget in self.inUseWidgets():
            widget.check()

    def getCheckedInUseWidgets(self):
        return [widget for widget in self.inUseWidgets() if widget.isChecked()]


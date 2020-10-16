from PyQt5.QtWidgets import  QWidget, QPushButton

class ExtendedQPushButton(QPushButton):
    def __init__(self, name, w = 100, h = 200):
        super(self.__class__, self).__init__()
        self.id = ''
        self.setText(name)
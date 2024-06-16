import sys

from PyQt5.QtWidgets import QApplication

from Classify import Classify
from upload_main import Window

if __name__ == '__main__':
    app = QApplication(sys.argv)
    classify = Classify()
    mywindow = Window(app, classify)
    mywindow.show()
    sys.exit(app.exec_())

import sys

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget

from Classify import Classify
from uploadimg import Ui_MainWindow


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, app, classify):
        super(QMainWindow, self).__init__()
        self.app = app
        self.setup_ui()  # 渲染画布
        self.connect_signals()  # 绑定触发事件
        self.classify = classify
        self.now_image = None
    def setup_ui(self):
        self.setupUi(self)

    def connect_signals(self):
        # 绑定触发事件
        self.btn_classify.clicked.connect(self.btn_classify_clicked)
        self.btn_select.clicked.connect(self.selectImage)
        self.history_list.itemClicked.connect(self.show_history)
    def btn_classify_clicked(self):
        # 检测功能（这里只是一个示例，实际功能需要根据具体需求实现）
        file_path = self.img_path.text()
        if file_path:
            # 显示加载动画或进度条
            self.progress_bar.setValue(50)
            # 检测按钮
            if self.method_combo.currentText() == "SVM":
                ans, predict = self.classify.ClassifyImage(self.now_image, self.classify.svclassifier_r)
                text1 = "换图概率:{}%".format(round(ans[0][0] * 100, 1))
                text2 = text1 + ",真人概率:{}%".format(round(ans[0][1] * 100, 1))
            elif self.method_combo.currentText() == "Logistic":
                ans, predict = self.classify.ClassifyImage(self.now_image, self.classify.logreg)
                text1 = "换图概率:{}%".format(round(ans[0][0] * 100, 1))
                text2 = text1 + ",真人概率:{}%".format(round(ans[0][1] * 100, 1))
            elif self.method_combo.currentText() == "CNN_f++":
                ans, predict = self.classify.calculate_accuracy(file_path)
                text1 = "换图概率:{}%".format(round(ans[0] * 100, 1))
                text2 = text1 + ",真人概率:{}%".format(round(predict[0] * 100, 1))
            elif self.method_combo.currentText() == "CNN_dfdc":
                ans, predict = self.classify.calculate_accuracy_dfdc_model(file_path)
                text1 = "换图概率:{}%".format(round(ans[0] * 100, 1))
                text2 = text1 + ",真人概率:{}%".format(round(predict[0] * 100, 1))
            self.result_show.setText(text2)
            print(predict[0])
            print("结果:", ans)
            self.progress_bar.setValue(100)
            result_text = "检测成功 - " + QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss") + " - "+self.method_combo.currentText() # 包含时间戳的结果
            self.add_to_history(self.img_path.text(), result_text,text2)

        else:
            self.result_show.setText("请先选择图片")


    def selectImage(self):
        image_path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Images (*.jpg *.png)')
        if image_path:
            self.img_path.setText(image_path)
            self.now_image = self.classify.read_pic_save_face(image_path)
            img_height, img_width, channels = self.now_image.shape
            bytesPerLine = channels * img_width
            QImg = QImage(self.now_image.data, img_width, img_height, bytesPerLine, QImage.Format_BGR888)
            self.img_show.setPixmap(QPixmap.fromImage(QImg))

    def add_to_history(self, img_path, result,text):
        # 添加新的检测结果到历史记录列表，并存储图片路径
        item = QtWidgets.QListWidgetItem()
        item.setText(result)
        item.img_path = img_path  # 将图片路径存储在列表项中
        item.text_show = text
        self.history_list.addItem(item)

    def show_history(self, item):
        # 当点击历史记录列表中的条目时，加载并显示图片，更新结果标签
        if item and hasattr(item, 'img_path'):
            self.img_path.setText(item.img_path)
            self.now_image = self.classify.read_pic_save_face(item.img_path)
            img_height, img_width, channels = self.now_image.shape
            bytesPerLine = channels * img_width
            QImg = QImage(self.now_image.data, img_width, img_height, bytesPerLine, QImage.Format_BGR888)
            self.img_show.setPixmap(QPixmap.fromImage(QImg))
            # 更新结果标签
            self.result_show.setText(item.text_show)

    def add_to_history_with_result(self, result):
        self.add_to_history(self.img_path.text(), result)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    classify = Classify()
    mywindow = Window(app, classify)
    mywindow.show()
    sys.exit(app.exec_())

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QProgressBar, QFileDialog, QListWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, \
    QLineEdit, QSplitter, QComboBox, QFrame


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)  # 增加窗口大小

        # 使用QSplitter进行分栏布局
        self.splitter = QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(2)
        self.splitter.setChildrenCollapsible(False)
        MainWindow.setCentralWidget(self.splitter)

        # 左侧布局，用于显示图片和控件
        self.left_widget = QtWidgets.QWidget(self.splitter)
        self.left_layout = QVBoxLayout(self.left_widget)

        # 图片显示区域
        self.img_show = QLabel(self.left_widget)
        self.img_show.setFixedSize(900, 450)  # 增加图片显示区域的大小
        self.img_show.setAlignment(QtCore.Qt.AlignCenter)
        self.left_layout.addWidget(self.img_show, alignment=QtCore.Qt.AlignHCenter)

        # 文件路径输入和按钮
        path_layout = QHBoxLayout()
        self.img_path = QLineEdit(self.left_widget)
        self.img_path.setMinimumWidth(300)  # 设置输入框的最小宽度
        path_layout.addWidget(self.img_path, stretch=1)  # 让输入框占据更多空间
        self.btn_select = QPushButton("选择图片", self.left_widget)
        self.btn_select.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        path_layout.addWidget(self.btn_select)
        # 添加方法选择框
        self.method_combo = QComboBox(self.left_widget)

        self.method_combo.addItem("SVM")  # 添加SVM选项
        self.method_combo.addItem("Logistic")  # 添加Logistic选项
        self.method_combo.addItem("CNN_f++")  # 添加CNN选项
        self.method_combo.addItem("CNN_dfdc")  # 添加CNN选项
        path_layout.addWidget(self.method_combo)

        self.left_layout.addLayout(path_layout)

        # 分类检测按钮
        self.btn_classify = QPushButton("开始检测", self.left_widget)
        self.btn_classify.setMinimumWidth(100)  # 设置按钮的最小宽度
        self.left_layout.addWidget(self.btn_classify, alignment=QtCore.Qt.AlignHCenter)
        # 检测结果展示
        result_frame = QFrame(self.left_widget)
        result_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        result_frame.setLineWidth(2)
        result_frame.setContentsMargins(10, 5, 10, 5)  # 设置内边距


        # 检测结果展示
        result_layout = QHBoxLayout(result_frame)
        self.result_label = QLabel("检测结果:", self.left_widget)
        self.result_label.setStyleSheet("font-weight: bold;")  # 设置字体加粗
        result_layout.addWidget(self.result_label)
        self.result_show = QLabel("", self.left_widget)
        self.result_show.setStyleSheet("""
                    font-size: 16px;
                    font-weight: bold;
                    color: #007bff;
                    padding: 5px;
                    border: 1px solid #007bff;
                    border-radius: 4px;
                    background-color: #f5f5f5;
                """)  # 设置样式
        result_layout.addWidget(self.result_show, stretch=1)  # 让结果标签占据更多空间
        self.left_layout.addWidget(result_frame, stretch=0)  # 添加到布局中，但不占据过多空间

        # 进度条
        self.progress_bar = QProgressBar(self.left_widget)
        self.progress_bar.setMinimumWidth(600)  # 设置进度条的最小宽度
        self.left_layout.addWidget(self.progress_bar, stretch=1)  # 让进度条占据更多空间

        # 右侧布局，用于显示历史记录
        self.right_widget = QtWidgets.QWidget(self.splitter)
        self.right_layout = QVBoxLayout(self.right_widget)
        self.history_list = QListWidget(self.right_widget)
        # 添加标题项
        self.history_title = QtWidgets.QListWidgetItem("历史记录")
        self.history_title.setBackground(QtGui.QColor("#f0f0f0"))  # 设置标题背景颜色
        self.history_title.setForeground(QtGui.QColor("#333333"))  # 设置标题文本颜色
        self.history_title.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))  # 设置标题字体
        self.history_list.addItem(self.history_title)  # 添加标题项

        # 设置列表样式
        self.history_list.setStyleSheet("""
            color: #333333;  /* 文本颜色 */
            background-color: #ffffff;  /* 背景颜色 */
            border: 1px solid #cccccc;  /* 边框 */
        """)
        # 禁止选择历史记录标题
        self.history_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.right_layout.addWidget(self.history_list, stretch=1)  # 让列表占据更多空间

        # 设置菜单栏和状态栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        # 重新翻译界面
        self.retranslateUi(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "深度伪造检测系统"))
        self.btn_select.setText(_translate("MainWindow", "选择图片"))
        self.btn_classify.setText(_translate("MainWindow", "开始检测"))
        self.result_label.setText(_translate("MainWindow", "检测结果:"))

        # 设置样式
        self.left_widget.setStyleSheet("""
            background-color: #f5f5f5;
            color: #333333;
            border-right: 1px solid #cccccc;
            padding: 10px;
        """)
        self.right_widget.setStyleSheet("""
            background-color: #ffffff;
            padding: 10px;
        """)

        self.btn_select.setStyleSheet("""
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            font-size: 14px;
        """)
        self.btn_select.setIconSize(QtCore.QSize(16, 16))

        self.btn_classify.setStyleSheet("""
            background-color: #28a745;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            font-size: 14px;
        """)

        self.progress_bar.setStyleSheet("""
            height: 20px;
        """)
        # 设置方法选择框的样式
        self.method_combo.setStyleSheet("""
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 2px;
            font-size: 14px;
            background-color: white;  /* 选择框背景颜色 */
        """)
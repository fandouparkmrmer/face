import sys
import cv2
import numpy as np
import os
import sqlite3
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, \
    QHBoxLayout, QLineEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtMultimedia import QCamera, QCameraImageCapture
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PIL import Image, ImageDraw, ImageFont

from train_model import Model, DataSet
from sqlite import init_db, add_to_blacklist, remove_from_blacklist, is_blacklisted


# 解决cv2.putText绘制中文乱码
def cv2ImgAddText(img2, text, left, top, textColor=(0, 0, 255), textSize=20):
    if isinstance(img2, np.ndarray):
        img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img2)
    fontStyle = ImageFont.truetype(r"C:\WINDOWS\FONTS\MSYH.TTC", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)




class CameraApp(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # 初始化数据库
        init_db()

        # 初始化相机和图像捕捉
        self.camera = QCamera()
        self.viewfinder = QCameraViewfinder(self)
        self.image_capture = QCameraImageCapture(self.camera)
        self.camera.setViewfinder(self.viewfinder)

        # 在初始化组件后调用 initUI
        self.initUI()

        # 设置定时器以周期性捕捉帧
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)
        self.timer.start(30)  # 每 30 毫秒更新一次

        self.camera.start()

        # 标志位，确保信号处理函数只调用一次
        self.saving_face_info = False
        self.recognizing_face = False

    def initUI(self):
        # 布局
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # 相机取景器
        self.viewfinder.setFixedSize(800, 600)
        self.viewfinder.setStyleSheet("border: 2px solid #000; border-radius: 5px;")
        left_layout.addWidget(self.viewfinder)

        # 姓名输入框
        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("请输入姓名")
        self.name_input.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.8);
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 5px;
        """)
        right_layout.addWidget(self.name_input)

        # 按钮
        self.save_button = QPushButton("保存人脸信息", self)
        self.save_button.clicked.connect(self.toggle_save_face_info)
        self.save_button.setStyleSheet("""
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
        """)
        right_layout.addWidget(self.save_button)

        self.detect_button = QPushButton("检测人脸信息", self)
        self.detect_button.clicked.connect(self.toggle_detect_face_info)
        self.detect_button.setStyleSheet("""
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
        """)
        right_layout.addWidget(self.detect_button)

        self.train_button = QPushButton("训练模型", self)
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setStyleSheet("""
            background-color: #f44336;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
        """)
        right_layout.addWidget(self.train_button)

        self.add_blacklist_button = QPushButton("添加至失信人员名单", self)
        self.add_blacklist_button.clicked.connect(self.add_to_blacklist)
        self.add_blacklist_button.setStyleSheet("""
            background-color: #FF9800;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
        """)
        right_layout.addWidget(self.add_blacklist_button)

        self.remove_blacklist_button = QPushButton("从失信人员名单删除", self)
        self.remove_blacklist_button.clicked.connect(self.remove_from_blacklist)
        self.remove_blacklist_button.setStyleSheet("""
            background-color: #FF5722;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
        """)
        right_layout.addWidget(self.remove_blacklist_button)

        # 文本输出框
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignTop)
        self.result_label.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.8);
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 5px;
            min-height: 100px;
        """)
        right_layout.addWidget(self.result_label)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def capture_frame(self):
        self.image_capture.capture()

    def toggle_save_face_info(self):
        if not self.saving_face_info:
            self.saving_face_info = True
            self.recognizing_face = False
            self.image_capture.imageCaptured.connect(self.process_face_info)
            self.save_button.setText("停止保存")
        else:
            self.saving_face_info = False
            self.image_capture.imageCaptured.disconnect(self.process_face_info)
            self.save_button.setText("保存人脸信息")

    def toggle_detect_face_info(self):
        if not self.recognizing_face:
            self.recognizing_face = True
            self.saving_face_info = False
            self.image_capture.imageCaptured.connect(self.process_face_info)
            self.detect_button.setText("停止检测")
        else:
            self.recognizing_face = False
            self.image_capture.imageCaptured.disconnect(self.process_face_info)
            self.detect_button.setText("检测人脸信息")

    def train_model(self):
        try:
            dataset = DataSet('dataset/')
            self.model.read_trainData(dataset)
            self.model.build_model()
            self.model.train_model()
            self.model.evaluate_model()
            self.model.save()
            self.result_label.setText("模型训练完成！")
        except Exception as e:
            self.result_label.setText(f"模型训练失败: {str(e)}")

    def process_face_info(self, id, image):
        try:
            image = image.convertToFormat(QImage.Format_RGB888)
            width = image.width()
            height = image.height()
            ptr = image.bits()
            ptr.setsize(height * width * 3)
            img = np.array(ptr).reshape(height, width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
            faces = face_cascade.detectMultiScale(gray, 1.35, 5)

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    R = gray[y:y + h, x:x + w]
                    R = cv2.resize(R, (128, 128), interpolation=cv2.INTER_LINEAR)

                    if self.recognizing_face:
                        label, prob = self.model.predict(R)
                        folder_names = os.listdir('dataset')
                        label_name = folder_names[label]

                        if prob > 0.7:
                            # 检查是否为失信人员
                            if is_blacklisted(label_name):
                                show_name = label_name
                                res = f"警告！！！检测为失信人员: {show_name}, 概率: {prob:.2f}"
                            else:
                                show_name = label_name
                                res = f"识别为: {show_name}, 概率: {prob:.2f}"
                        else:
                            show_name = "陌生人"
                            res = "抱歉，未识别出该人！请尝试增加数据量来训练模型！"

                        img = cv2ImgAddText(img, show_name, x + 5, y - 30)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    elif self.saving_face_info:
                        name = self.name_input.text().strip()
                        if name:
                            # Create directories for storing images and models
                            data_dir = os.path.join('data', name)
                            dataset_dir = os.path.join('dataset', name)
                            os.makedirs(data_dir, exist_ok=True)
                            os.makedirs(dataset_dir, exist_ok=True)

                            # Save the raw captured face image
                            image_path = os.path.join(data_dir, f'{name}_{len(os.listdir(data_dir))}.jpg')
                            cv2.imwrite(image_path, img)

                            # Save the grayscale image for model training
                            model_image_path = os.path.join(dataset_dir, f'{name}_{len(os.listdir(dataset_dir))}.jpg')
                            cv2.imwrite(model_image_path, R)

                            res = f"{name} 的人脸信息已保存。"
                        else:
                            res = "请输入姓名。"

                    self.result_label.setText(res)
            else:
                self.result_label.setText("未检测到人脸")
        except Exception as e:
            self.result_label.setText(f"处理人脸信息失败: {str(e)}")

    def add_to_blacklist(self):
        name = self.name_input.text().strip()
        if name:
            add_to_blacklist(name)
            self.result_label.setText(f"{name} 已添加至失信人员名单。")
        else:
            self.result_label.setText("请输入姓名。")

    def remove_from_blacklist(self):
        name = self.name_input.text().strip()
        if name:
            remove_from_blacklist(name)
            self.result_label.setText(f"{name} 已从失信人员名单中删除。")
        else:
            self.result_label.setText("请输入姓名。")



class FaceDetectionApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("人脸检测应用")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.upload_button = QPushButton("图片识别")
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setFixedSize(779, 50)

        self.camera_button = QPushButton("摄像头识别")
        self.camera_button.clicked.connect(self.start_camera_detection)
        self.camera_button.setFixedSize(779, 50)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(800, 500)

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.camera_button)

        self.central_widget.setLayout(self.layout)

        # 加载模型
        self.model = Model()
        self.model.load()

    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择图像文件", "", "图像文件 (*.png *.jpg *.jpeg)")

        if file_path:
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
            faces = face_cascade.detectMultiScale(gray, 1.35, 5)

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    R = gray[y:y + h, x:x + w]
                    R = cv2.resize(R, (128, 128), interpolation=cv2.INTER_LINEAR)

                    label, prob = self.model.predict(R)
                    folder_names = os.listdir('dataset')
                    label_name = folder_names[label]

                    if prob > 0.7:
                        show_name = label_name
                        res = f"识别为: {show_name}, 概率: {prob:.2f}"
                    else:
                        show_name = "陌生人"
                        res = "抱歉，未识别出该人！请尝试增加数据量来训练模型！"

                    image = cv2ImgAddText(image, show_name, x + 5, y - 30)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

                self.image_label.setPixmap(self.convert_cv_qt(image, 800, 500))
                self.image_label.setText(res)
            else:
                self.image_label.setText("未检测到人脸")

    def convert_cv_qt(self, cv_img, width, height):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def start_camera_detection(self):
        self.camera_widget = CameraApp(self.model)
        self.setCentralWidget(self.camera_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FaceDetectionApp()
    main_window.show()
    sys.exit(app.exec_())

import pickle

import cv2
import numpy as np
from scipy.interpolate import griddata
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from PIL import Image
import tensorflow as tf
import radialProfile


class Classify:
    model = tf.keras.models.load_model('./DeepFakeDetection/save/model')
    dfdc_model = tf.keras.models.load_model('./DeepFakeDetection/save/DFDCmodel')
    def __init__(self):
        self.svclassifier_r,self.logreg = self.train()
    def train(self):
        #train
        pkl_file = open('train_3200.pkl', 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        X = data["data"]
        y = data["label"]
        # 创建一个SVC分类器实例，并设置其参数。C是正则化参数，kernel指定核函数类型为径向基函数（RBF），gamma是核函数的参数。
        svclassifier_r = SVC(C=6.37, kernel='rbf', gamma=0.86,probability=True)
        # 使用训练数据集X和标签y来训练SVC分类器
        svclassifier_r.fit(X, y)
        # 创建一个逻辑回归模型实例，并设置参数。solver指定了用于优化的算法，liblinear是一个适合小数据集的算法，max_iter是最大迭代次数。
        logreg = LogisticRegression(solver='liblinear', max_iter=1000)
        # 使用训练数据集X和标签y来训练逻辑回归模型。
        logreg.fit(X, y)
        return svclassifier_r,logreg

    def ClassifyImageByfileName(self,fileName,method):
        N = 300  # N定义了功率谱密度(PSD)的采样点数。
        number_iter = 1  # number_iter定义了迭代次数，即处理的图像数量。
        psd1D_total = np.zeros([number_iter, N])  # 存储所有图像的PSD的数组
        with open(fileName, 'rb') as file:
            # 读取文件内容到一个字节数组
            file_bytes = file.read()
        # 将字节数组转换为NumPy数组
        image_array = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(image_array,0)
        # 裁剪图像，保留图像三分之一
        h = int(img.shape[0] / 3)
        w = int(img.shape[1] / 3)
        img = img[h:-h, w:-w]
        f = np.fft.fft2(img)  # 进行二维快速傅里叶变换（FFT），得到频域表示 f
        fshift = np.fft.fftshift(f)  # 对 FFT 结果进行频谱中心化，将零频分量移动到频谱的中心。
        magnitude_spectrum = 20 * np.log(np.abs(fshift))  # 计算幅度谱，并取其对数得到 magnitude_spectrum
        psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)  # 计算 magnitude_spectrum 的一维径向平均功率谱密度（PSD）
        points = np.linspace(0, N, num=psd1D.size)
        xi = np.linspace(0, N, num=N)
        interpolated = griddata(points, psd1D, xi,
                                method='cubic')  # 为了在固定的频率点上进行分析，使用 griddata 函数对 PSD 进行插值，得到 interpolated。
        interpolated /= interpolated[0]  # 将插值后的 PSD 除以其第一个元素进行归一化。
        psd1D_total[0, :] = interpolated  # 将归一化后的 PSD 存储到 psd1D_total 数组中
        ans = method.predict_proba(psd1D_total)
        ans_predict = method.predict(psd1D_total)
        print("predict:",ans_predict)
        return ans,ans_predict
        # print("logreg:",self.logreg.predict_proba(psd1D_total))
    def ClassifyImage(self,img,method):
        N = 300  # N定义了功率谱密度(PSD)的采样点数。
        number_iter = 1  # number_iter定义了迭代次数，即处理的图像数量。
        psd1D_total = np.zeros([number_iter, N])  # 存储所有图像的PSD的数组
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 裁剪图像，保留图像三分之一
        h = int(img.shape[0] / 3)
        w = int(img.shape[1] / 3)
        img = img[h:-h, w:-w]
        f = np.fft.fft2(img)  # 进行二维快速傅里叶变换（FFT），得到频域表示 f
        fshift = np.fft.fftshift(f)  # 对 FFT 结果进行频谱中心化，将零频分量移动到频谱的中心。
        magnitude_spectrum = 20 * np.log(np.abs(fshift))  # 计算幅度谱，并取其对数得到 magnitude_spectrum
        psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)  # 计算 magnitude_spectrum 的一维径向平均功率谱密度（PSD）
        points = np.linspace(0, N, num=psd1D.size)
        xi = np.linspace(0, N, num=N)
        interpolated = griddata(points, psd1D, xi,
                                method='cubic')  # 为了在固定的频率点上进行分析，使用 griddata 函数对 PSD 进行插值，得到 interpolated。
        interpolated /= interpolated[0]  # 将插值后的 PSD 除以其第一个元素进行归一化。
        psd1D_total[0, :] = interpolated  # 将归一化后的 PSD 存储到 psd1D_total 数组中
        ans = method.predict_proba(psd1D_total)
        ans_predict = method.predict(psd1D_total)
        print("predict:",ans_predict)
        return ans,ans_predict
        # print("logreg:",self.logreg.predict_proba(psd1D_total))
    def read_pic_save_face(self,sourcePath):
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
        img = cv2.imread(sourcePath)
        if type(img) != str:
            faces = face_cascade.detectMultiScale(img, 1.1, 5)
        if len(faces):
            return self.write_face(faces, img)
    def write_face(self, faces, img):
        for (x, y, w, h) in faces:
            if w >= 16 and h >= 16:
                # 扩大图片，可根据坐标调整
                X = int(x)
                W = min(int(x + w), img.shape[1])
                Y = int(y)
                H = min(int(y + h), img.shape[0])
                f = cv2.resize(img[Y:H, X:W], (W - X, H - Y))
        return f

    def load_and_process_image(self,image_path, target_size=(128, 128)):
        image = Image.open(image_path).convert('RGB')
        resized_image = image.resize(target_size)
        image_array = np.array(resized_image) / 255
        return image_array

    def check_and_adjust_image(self,image, target_size=(128, 128, 3)):
        if image.shape == target_size:
            return image
        else:
            pil_image = Image.fromarray(image)
            resized_image = pil_image.resize(target_size)
            return np.array(resized_image)

    def calculate_accuracy(self,image_path):
        validate_array = []
        image_array = self.load_and_process_image(image_path)
        adjusted_image = self.check_and_adjust_image(image_array)
        validate_array.append(adjusted_image)

        validate_array = np.array(validate_array)
        predictions = self.model.predict(validate_array)

        return 1 - predictions[0], predictions[0]
    def calculate_accuracy_dfdc_model(self,image_path):
        validate_array = []
        image_array = self.load_and_process_image(image_path)
        adjusted_image = self.check_and_adjust_image(image_array)
        validate_array.append(adjusted_image)

        validate_array = np.array(validate_array)
        predictions = self.dfdc_model.predict(validate_array)

        return 1 - predictions[0], predictions[0]
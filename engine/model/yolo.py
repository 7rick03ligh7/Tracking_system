import tensorflow as tf
import numpy as np
import cv2 as cv
import time
import sys
import matplotlib.pyplot as plt
import engine.model.layers as layers
import os


class Yolo:
    """
    Class for define YOLO neural network for object recognition
    """

    alpha = 0.1
    threshold = 0.2
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]

    w_img = 640
    h_img = 480

    def __init__(self, yolo_type: str):
        """
        Create YOLO instance

        Arguments:
            yolo_type {str} -- type Yolo weight ('small', 'tiny')
        """

        self.yolo_type = yolo_type
        self.last_layer = None
        if yolo_type == 'small':
            self.weights_file = os.getcwd() + '/weights/yolo/YOLO_small.ckpt'
        elif yolo_type == 'tiny':
            self.weights_file = os.getcwd() + '/weights/yolo/YOLO_tiny.ckpt'
        else:
            return
        print(self.weights_file)
        self.build_networks()

    def build_networks(self):
        """
        Construct architecture and load weight
        """

        if self.yolo_type == 'small':
            self.x = tf.placeholder('float32', [None, 448, 448, 3])
            self.conv_1 = layers.conv_layer(
                1, self.alpha, self.x, 64, 7, 2)
            self.pool_2 = layers.pooling_layer(2, self.conv_1, 2, 2)
            self.conv_3 = layers.conv_layer(
                3, self.alpha, self.pool_2, 192, 3, 1)
            self.pool_4 = layers.pooling_layer(4, self.conv_3, 2, 2)
            self.conv_5 = layers.conv_layer(
                5, self.alpha, self.pool_4, 128, 1, 1)
            self.conv_6 = layers.conv_layer(
                6, self.alpha, self.conv_5, 256, 3, 1)
            self.conv_7 = layers.conv_layer(
                7, self.alpha, self.conv_6, 256, 1, 1)
            self.conv_8 = layers.conv_layer(
                8, self.alpha, self.conv_7, 512, 3, 1)
            self.pool_9 = layers.pooling_layer(9, self.conv_8, 2, 2)
            self.conv_10 = layers.conv_layer(
                10, self.alpha, self.pool_9, 256, 1, 1)
            self.conv_11 = layers.conv_layer(
                11, self.alpha, self.conv_10, 512, 3, 1)
            self.conv_12 = layers.conv_layer(
                12, self.alpha, self.conv_11, 256, 1, 1)
            self.conv_13 = layers.conv_layer(
                13, self.alpha, self.conv_12, 512, 3, 1)
            self.conv_14 = layers.conv_layer(
                14, self.alpha, self.conv_13, 256, 1, 1)
            self.conv_15 = layers.conv_layer(
                15, self.alpha, self.conv_14, 512, 3, 1)
            self.conv_16 = layers.conv_layer(
                16, self.alpha, self.conv_15, 256, 1, 1)
            self.conv_17 = layers.conv_layer(
                17, self.alpha, self.conv_16, 512, 3, 1)
            self.conv_18 = layers.conv_layer(
                18, self.alpha, self.conv_17, 512, 1, 1)
            self.conv_19 = layers.conv_layer(
                19, self.alpha, self.conv_18, 1024, 3, 1)
            self.pool_20 = layers.pooling_layer(20, self.conv_19, 2, 2)
            self.conv_21 = layers.conv_layer(
                21, self.alpha, self.pool_20, 512, 1, 1)
            self.conv_22 = layers.conv_layer(
                22, self.alpha, self.conv_21, 1024, 3, 1)
            self.conv_23 = layers.conv_layer(
                23, self.alpha, self.conv_22, 512, 1, 1)
            self.conv_24 = layers.conv_layer(
                24, self.alpha, self.conv_23, 1024, 3, 1)
            self.conv_25 = layers.conv_layer(
                25, self.alpha, self.conv_24, 1024, 3, 1)
            self.conv_26 = layers.conv_layer(
                26, self.alpha, self.conv_25, 1024, 3, 2)
            self.conv_27 = layers.conv_layer(
                27, self.alpha, self.conv_26, 1024, 3, 1)
            self.conv_28 = layers.conv_layer(
                28, self.alpha, self.conv_27, 1024, 3, 1)
            self.fc_29 = layers.fc_layer(
                29, self.alpha, self.conv_28, 512, flat=True, linear=False)
            self.fc_30 = layers.fc_layer(
                30, self.alpha, self.fc_29, 4096, flat=False, linear=False)
            # skip dropout_31
            self.fc_32 = layers.fc_layer(
                32, self.alpha, self.fc_30, 1470, flat=False, linear=True)
            self.last_layer = self.fc_32
        if self.yolo_type == 'tiny':
            self.x = tf.placeholder('float32', [None, 448, 448, 3])
            self.conv_1 = layers.conv_layer(1, self.alpha, self.x, 16, 3, 1)
            self.pool_2 = layers.pooling_layer(2, self.conv_1, 2, 2)
            self.conv_3 = layers.conv_layer(
                3, self.alpha, self.pool_2, 32, 3, 1)
            self.pool_4 = layers.pooling_layer(4, self.conv_3, 2, 2)
            self.conv_5 = layers.conv_layer(
                5, self.alpha, self.pool_4, 64, 3, 1)
            self.pool_6 = layers.pooling_layer(6, self.conv_5, 2, 2)
            self.conv_7 = layers.conv_layer(
                7, self.alpha, self.pool_6, 128, 3, 1)
            self.pool_8 = layers.pooling_layer(8, self.conv_7, 2, 2)
            self.conv_9 = layers.conv_layer(
                9, self.alpha, self.pool_8, 256, 3, 1)
            self.pool_10 = layers.pooling_layer(10, self.conv_9, 2, 2)
            self.conv_11 = layers.conv_layer(
                11, self.alpha, self.pool_10, 512, 3, 1)
            self.pool_12 = layers.pooling_layer(12, self.conv_11, 2, 2)
            self.conv_13 = layers.conv_layer(
                13, self.alpha, self.pool_12, 1024, 3, 1)
            self.conv_14 = layers.conv_layer(
                14, self.alpha, self.conv_13, 1024, 3, 1)
            self.conv_15 = layers.conv_layer(
                15, self.alpha, self.conv_14, 1024, 3, 1)
            self.fc_16 = layers.fc_layer(
                16, self.alpha, self.conv_15, 256, flat=True, linear=False)
            self.fc_17 = layers.fc_layer(
                17, self.alpha, self.fc_16, 4096, flat=False, linear=False)
            # skip dropout_18
            self.fc_19 = layers.fc_layer(
                19, self.alpha, self.fc_17, 1470, flat=False, linear=True)
            self.last_layer = self.fc_19
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def detect(self, img: np.ndarray, cvmat: bool=False) -> list:
        """
        Recognizing objects in image(frame)

        Arguments:
            img {np.ndarray} -- image for recognizing

        Keyword Arguments:
            cvmat {bool} -- if True input img is cvmat type (default: {False})

        Returns:
            list -- result of recognition
        """

        s = time.time()
        self.h_img, self.w_img, _ = img.shape
        img_resized = cv.resize(img, (448, 448))
        if cvmat:
            img_RGB = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)
        else:
            img_RGB = img_resized
        img_resized_np = np.asarray(img_RGB)
        inputs = np.zeros((1, 448, 448, 3), dtype='float32')
        inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        in_dict = {self.x: inputs}
        net_output = self.sess.run(self.last_layer, feed_dict=in_dict)
        strtime = str(time.time() - s)
        self.result = self.interpret_output(net_output[0])
        return self.result

    def detect_from_file(self, filename: str) -> list:
        """
        Useless function for recognizing object from file

        Arguments:
            filename {str} -- path for image

        Returns:
            list -- result of recognition
        """

        img = cv.imread(filename)
        self.detect(img)
        return self.result

    def interpret_output(self, output: np.ndarray) -> iter:
        """
        Inner YOLO post-processing

        Arguments:
            output {np.ndarray} -- inner output of YOLO

        Returns:
            list -- bbox yolo type
        """

        probs = np.zeros((7, 7, 2, 20))
        class_probs = np.reshape(output[0:980], (7, 7, 20))
        scales = np.reshape(output[980:1078], (7, 7, 2))
        boxes = np.reshape(output[1078:], (7, 7, 2, 4))
        offset = np.transpose(np.reshape(
            np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
        boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
        boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

        boxes[:, :, :, 0] *= self.w_img
        boxes[:, :, :, 1] *= self.h_img
        boxes[:, :, :, 2] *= self.w_img
        boxes[:, :, :, 3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[:: -1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i],
                            boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]],
                           boxes_filtered[i][0], boxes_filtered[i][1],
                           boxes_filtered[i][2], boxes_filtered[i][3],
                           probs_filtered[i]])

        return result

    def iou(self, box1: iter, box2: iter) -> float:
        """
        Calculating intersection over union (IOU) between two bbox yolo type

        Arguments:
            box1 {iter} -- bbox yolo type
            box2 {iter} -- bbox yolo type

        Returns:
            float -- IOU result
        """

        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] +
                               box2[2] * box2[3] - intersection)

"""
This is detection file that will be used to detect person bounding box with
the confident score.

Public function:
    - detect_person(): return bounding box, score for person
    - detect_person_and_plot_to_output_image(): plot output image

Commands:
    python src/yolov3/detector.py
    python src/yolov3/detector.py --tiny
"""
import os
import yaml

import numpy as np
from absl import logging
import tensorflow as tf

from .graph_functions import load_graph
from .dataset import transform_images


class Detector:

    def __init__(self, config):
        self.config = config
        self.root_dir = config['root']

        self.yolo = self._create_yolo_model()

    @staticmethod
    def _setup_gpu():
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            logging.info('GPU setup with %d devices', len(physical_devices))
        else:
            logging.info('use CPU')

    def _create_yolo_model(self):
        '''
        Creates yolo model for human detection
        '''
        model_type = self.config['model_type']
        if self.config['yolo_graph_mode']:
            model_path = os.path.join(self.root_dir, self.config['graph_files'][model_type])
            return self._load_yolo_graph(model_path)
        model_path = os.path.join(self.root_dir, self.config['model_files'][model_type])
        model = tf.keras.models.load_model(model_path)
        return model

    def _load_yolo_graph(self, filepath):
        '''
        When loading a graph model, you need to explicitly state the input
        and output nodes of the graph. It is usually x:0 for input and Identity:0
        for outputs, depending on how many output nodes you have.
        '''
        model_type = 'yolo%s' % self.config['model_type'][:2]
        model_nodes = self.config['MODEL_NODES'][model_type]
        model_path = os.path.join(filepath)
        if os.path.isfile(model_path):
            return load_graph(model_path, inputs=model_nodes['inputs'],
                            outputs=model_nodes['outputs'])
        raise ValueError('Graph file does not exist. Please check that '
                        '%s exists' % model_path)

    @staticmethod
    def _load_image(image_file):
        img = open(image_file, 'rb').read()
        logging.info('image file %s loaded', image_file)
        return img

    @staticmethod
    def _reshape_image(image, image_size):
        image = tf.expand_dims(image, 0)
        image = transform_images(image, image_size)
        return image

    @staticmethod
    def _shrink_dimension_and_length(boxes, scores, classes, nums, object_ids=[0]):
        len0 = nums[0]

        classes = classes.numpy()[0]
        classes = classes[:len0]
        mask1 = np.isin(classes, tuple(object_ids))  # only identify objects we are interested in
        classes = tf.boolean_mask(classes, mask1)

        scores = scores.numpy()[0]
        scores = scores[:len0]
        scores = tf.boolean_mask(scores, mask1)

        boxes = boxes.numpy()[0]
        boxes = boxes[:len0]
        boxes = tf.boolean_mask(boxes, mask1)

        return boxes, scores, classes

    def _evaluate_image_by_yolo(self, image):
        '''
        Takes in the yolo model and image to perform inference with.
        It will return the following:
            - boxes: the bounding boxes for each object
            - scores: the scores for each object predicted
            - classes: the class predicted for each bounding box
            - nums: number of valid bboxes. Only nums[0] should be used. The rest
                    are paddings.
        '''
        # image = image[..., ::-1]  # swap from bgr to rgb
        pred = self.yolo(image)[-1]
        bboxes = pred[:, :, :4].numpy()
        bboxes[:, :, [0, 1]] = bboxes[:, :, [1, 0]]  # swapping x and y axes
        bboxes[:, :, [2, 3]] = bboxes[:, :, [3, 2]]
        pred_conf = pred[:, :, 4:]

        # performs nms using model's predictions
        boxes, scores, classes, nums = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bboxes, (tf.shape(bboxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.config['yolo_iou_threshold'],
            score_threshold=self.config['yolo_score_threshold']
        )
        return boxes, scores, classes, nums

    @staticmethod
    def _prepare_image_from_camera(image):
        image = image.astype(np.float32)
        image = tf.convert_to_tensor(image)
        return image

    @staticmethod
    def _prepare_image_from_file(image):
        image = tf.image.decode_image(image, channels=3)
        return image

    def predict_person_bbox_from_image(self, image):
        """Detect all persons' bounding box from one image

        args:
            - yolo:  (Model) model like yolov3 or yolov3_tiny
            - image: (np.array) input image

        return:
            - boxes: (np.array) an array of bounding box with
                    definition like (x1, y1, x2, y2), in a
                    coordinate system with original point in
                    the left top corner
        """
        # 1. prepare image
        image = self._prepare_image_from_camera(image)
        image = self._reshape_image(image, image_size=self.config['size'])

        # 2. evaluate image
        boxes, scores, classes, nums = self._evaluate_image_by_yolo(image)

        # 3. clean up return
        boxes, scores, classes = self._shrink_dimension_and_length(
            boxes, scores, classes, nums)

        boxes = np.array(boxes)
        return boxes

    # possible that we may want to control what is being detection
    def predict_object_bbox_from_image(self, class_names, image, detect_ids):
        """Detect all objects' bounding box from one image

        args:
            - yolo:  (Model) model like yolov3 or yolov3_tiny
            - image: (np.array) input image

        return:
            - boxes: (np.array) an array of bounding box with
                    definition like (x1, y1, x2, y2), in a
                    coordinate system with original point in
                    the left top corner
        """
        # 1. prepare image
        image = self._prepare_image_from_camera(image)
        image = self._reshape_image(image, self.config['size'])

        # 2. evaluate image
        boxes, scores, classes, nums = self._evaluate_image_by_yolo(image)

        # 3. clean up return
        boxes, scores, classes = self._shrink_dimension_and_length(
            boxes, scores, classes, nums, detect_ids)

        # convert classes into class names
        classes = [class_names[int(i)] for i in classes]

        boxes = np.array(boxes)
        return boxes, classes, scores

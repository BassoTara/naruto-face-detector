import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import glob
import cv2
from darkflow.net.build import TFNet
from pandas import *
import numpy as np
import copy
import configparser


# Loads the specified config file and returns its values
def load_config(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    config.sections()

    model_path = ConfigSectionMap(config, 'TFNet')['model_relative_path']
    min_steps = int(ConfigSectionMap(config, 'TFNet')['load_min_steps'])
    max_steps = int(ConfigSectionMap(config, 'TFNet')['load_max_steps'])
    increment = int(ConfigSectionMap(config, 'TFNet')['load_increment'])
    accuracy_threshold = float(ConfigSectionMap(config, 'TFNet')['accuracy_threshold'])
    gpu_usage = float(ConfigSectionMap(config, 'TFNet')['gpu_usage'])
    validation_xml_path = ConfigSectionMap(config, 'ValidationSet')['annotations_xml_path']
    iou_threshold = float(ConfigSectionMap(config, 'IoU')['iou_threshold'])
    weight_matrix_path = ConfigSectionMap(config, 'ConfusionMatrix')['weight_matrix_path']

    return model_path, min_steps, max_steps, increment, accuracy_threshold, \
           gpu_usage, validation_xml_path, iou_threshold, weight_matrix_path


# Helper function for load_config
def ConfigSectionMap(config, section):
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option] = config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1


# Reads the labels.txt file and returns its content as a list
def read_labels():
    with open('labels.txt') as f:
        labels = f.readlines()
    labels.append('none')
    labels = [x.strip() for x in labels]

    # Print for debug
    # print('Labels = ', labels)

    return labels


# Initializes the TFNet
def init_net(model_path, load, accuracy_threshold, gpu_usage):
    options = {'model': model_path,
               'load': load,
               'threshold': accuracy_threshold,
               'gpu': gpu_usage
               }

    tfnet = TFNet(options)
    return tfnet


# Parses the annotation file and returns a list of objects
# and the image file loaded from the specified path
def parse_annotation(filename):
    doc = ET.parse(filename)

    objects = doc.findall('object')

    path = doc.find('path').text
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return objects, img


# Creates and returns weight matrix.
# Edit this code if you don't want to use the identity matrix
def load_weight_matrix(weight_matrix_path):
    weight_matrix = np.loadtxt(weight_matrix_path, dtype=float)
    return weight_matrix


# Computes the confusion matrix iterating through predicted and ground truth bounding boxes
def compute_confusion_matrix(confusion_matrix_dictionary, results, objects, iou_threshold):
    # Detects matches between ground truth and predicted bounding boxes basing on IoU values.
    # Matching labels will be used as index in order to increment the confusion matrix values.

    # Create a copy of objects in order to edit a list without affecting the other one
    tmp_objects = copy.copy(objects)

    for result in results:
        box_d = results_bounding_box(result)
        match_found = False
        for object in objects:
            box_gt = annotations_bounding_box(object)
            precision = bb_intersection_over_union(box_gt, box_d)
            if precision > iou_threshold:
                confusion_matrix_dictionary = \
                    update_dictionary(confusion_matrix_dictionary, object, result)
                match_found = True
                if object in tmp_objects:
                    tmp_objects.remove(object)

        if not match_found:
            confusion_matrix_dictionary = \
                update_dictionary(confusion_matrix_dictionary, None, result)

    for object in tmp_objects:
        confusion_matrix_dictionary = \
            update_dictionary(confusion_matrix_dictionary, object, None)

    return confusion_matrix_dictionary


# Checks if two rectangles intersect each other and returns their IoU value
def bb_intersection_over_union(boxA, boxB):
    iou = 0

    if not (boxA[2] < boxB[0] or boxA[0] > boxB[2] or boxA[1] > boxB[3] or boxA[3] < boxB[1]):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])

        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        inter_area = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        box_b_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou


# Shows the image with predicted bounding boxes
def draw_boxes(results, img):
    # Generate random colors
    colors = [tuple(255 * np.random.rand(3)) for i in results]

    # Iterate on a list of couples (color, result)
    for color, result in zip(colors, results):
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        # confidence = result['confidence']

        # Add the box and label...
        img = cv2.rectangle(img, tl, br, color, 5)
        img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    # ...and display it
    plt.imshow(img)
    plt.show()


# Creates an array containing the values of the predicted bounding box
def results_bounding_box(result):
    box_d = [result['topleft']['x'], result['topleft']['y'],
             result['bottomright']['x'], result['bottomright']['y']]
    # print('Pred BB = ', box_d)
    return box_d


# Creates an array containing the values of the ground truth bounding box
def annotations_bounding_box(object):
    tlx = int(object.find('bndbox/xmin').text)
    tly = int(object.find('bndbox/ymin').text)
    brx = int(object.find('bndbox/xmax').text)
    bry = int(object.find('bndbox/ymax').text)
    box_g_t = [tlx, tly, brx, bry]
    # print('GT BB = ', box_g_t)
    return box_g_t


# Updates the confusion matrix with the specified values and returns it
def update_dictionary(confusion_matrix_dictionary, object, result):
    row = 'none'
    col = 'none'

    if object is not None:
        row = object.find('name').text

    if result is not None:
        col = result['label']

    confusion_matrix_dictionary[row][col] += 1
    # print(col, "(predicted) matched with (GT)", row)
    return confusion_matrix_dictionary


# Initializes the writing process to store results and returns the textfile
def init_print(labels, accuracy_threshold):
    file = open(''.join(['results/results_labels_', str(len(labels) - 1), '.txt']), 'w')
    file.write(''.join(['Labels = ', str(labels), '\n']))
    file.write(''.join(['Accuracy Threshold = ', str(accuracy_threshold), '\n\n']))

    return file


# Stores results into the specified textfile
def print_results(file, load, correct_matches, total_matches):
    file.write(''.join(['Load = ', str(load), '\n']))
    file.write(''.join(['Correct matches = ', str(correct_matches), '\n']))
    file.write(''.join(['Total matches = ', str(total_matches), '\n']))

    if total_matches is not 0:
        file.write(''.join(['Precision = ', str(correct_matches / total_matches), '\n\n']))
    else:
        file.write('No matches found\n\n')


# Weights the confusion matrix using the weight matrix and returns the sum of
# the weighted elements and the total number of elements
def sum_confusion_matrix(confusion_matrix_dictionary, labels, weight_matrix):
    df = DataFrame(confusion_matrix_dictionary, index=labels,
                   columns=labels).T

    # Print for debug
    print(df)

    confusion_matrix = df.as_matrix()

    weighted_confusion_matrix = np.multiply(confusion_matrix, weight_matrix)

    return weighted_confusion_matrix.sum(), confusion_matrix.sum()


# Creates a 2D dictionary using the specified labels in both dimensions
def create_2d_dictionary(labels):
    d = {}

    for label1 in labels:
        d[label1] = {}
        for label2 in labels:
            d[label1][label2] = 0

    # Print for debug
    # print(d)

    return d


model_path, min_steps, max_steps, increment, accuracy_threshold, \
gpu_usage, validation_xml_path, iou_threshold, weight_matrix_path = load_config(
    'mangafacesconfig.ini')
labels = read_labels()

text_file = init_print(labels, accuracy_threshold)

weight_matrix = load_weight_matrix(weight_matrix_path)

for load in range(min_steps, max_steps + 1, increment):

    tfnet = init_net(model_path, load, accuracy_threshold, gpu_usage)

    correct_matches = total_matches = 0

    # Loops over the annotation files of validation set
    for filename in glob.glob(validation_xml_path):
        objects, img = parse_annotation(filename)

        # Uses YOLO to predict the image
        results = tfnet.return_predict(img)

        confusion_matrix_dictionary = create_2d_dictionary(labels)

        confusion_matrix_dictionary = compute_confusion_matrix(
            confusion_matrix_dictionary, results, objects, iou_threshold)

        current_correct_matches, current_total_matches = \
            sum_confusion_matrix(confusion_matrix_dictionary, labels, weight_matrix)

        correct_matches += current_correct_matches
        total_matches += current_total_matches

        # Uncomment the line below if you want to show the images
        # draw_boxes(results, img)

    print_results(text_file, load, correct_matches, total_matches)

text_file.close()

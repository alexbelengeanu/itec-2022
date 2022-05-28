from collections import OrderedDict
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
import cv2
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from consts import ANGLE, START_ANGLE, END_ANGLE, RED, GREEN, BLUE, NUM_OF_SAMPLES

def generate_blank() -> np.ndarray:
    """
    Function used to generate the blank image of size 96x96.
    :return: Blank image as numpy array.
    """
    return np.ones(shape=(96,96,3), dtype=np.int16)

def draw_random_rectangle(image: Image, 
                          color: Tuple) -> np.ndarray:
    """
    Function used to randomly draw a rectangle on a black image.
    :param image: Blank image to draw the rectangle on.
    :param color: Tuple representing the image code in RGB format.
    :return: np.ndarray representing the image after drawing.
    """
    image_ = image.copy()
    x_1 = x_2 = y_1 = y_2 = 0
    while x_1 - x_2 < 25 or y_1 - y_2 < 25:
        x_1 = random.randint(10, 86)
        y_1 = random.randint(10, 86)
        x_2 = random.randint(10, 86)
        y_2 = random.randint(10, 86)
    width = np.max(np.array([x_1, x_2])) - np.min(np.array([x_1, x_2]))
    height = np.max(np.array([y_1, y_2])) - np.min(np.array([y_1, y_2]))
    area = width * height
    return cv2.rectangle(image_, pt1=(x_1,y_1), pt2=(x_2,y_2), color=color, thickness= -1), ((100 * area) / 9216) / 100

def generate_random_triangle_point() -> int:
    """
    Function used to randomly generate points for triangles, based on some rules I tested on paper.
    :return: Integer representing point coordinate on x/y axis.
    """
    first_interval = np.array([0.1, 0.4])
    second_interval = np.array([0.56, 0.86])

    total_length = np.ptp(first_interval)+np.ptp(second_interval)
    n = 1
    numbers = np.random.random(n)*total_length
    numbers += first_interval.min()
    numbers[numbers > first_interval.max()] += second_interval.min()-first_interval.max()
    numbers = np.floor(numbers * 100).astype(int)
    return numbers[0]

def draw_random_triangle(image: Image, 
                         color: Tuple) -> np.ndarray:
    """
    Function used to randomly draw a triangle on a black image.
    :param image: Blank image to draw the triangle on.
    :param color: Tuple representing the image code in RGB format.
    :return: np.ndarray representing the image after drawing.
    """
    image_ = image.copy()
    area = 0

    # short explanation for the snippet below
    # After trial&error I found out that if area < 800 and difference between points is <= 15px, chances that the triangle looks like
    # a line are very high. So I am restricting the triangle to have an area > 800 and points to have a larger distance than 15px

    while area < 800 : 
        x_1 = x_2 = x_3 = y_1 = y_2 = y_3 = 10
        while x_1 - x_2 < 15:
            x_1 = generate_random_triangle_point()
            x_2 = generate_random_triangle_point()
            while x_1 - x_3 < 15:
                x_1 = generate_random_triangle_point()
                x_3 = generate_random_triangle_point()
                while x_2 - x_3 < 15:
                    x_2 = generate_random_triangle_point()
                    x_3 = generate_random_triangle_point()
                    while y_1 - y_2 < 15:
                        y_1 = generate_random_triangle_point()
                        y_2 = generate_random_triangle_point()
                        while y_1 - y_3 < 15:
                            y_1 = generate_random_triangle_point()
                            y_3 = generate_random_triangle_point()
                            while y_2 - y_3 < 15:
                                y_2 = generate_random_triangle_point()
                                y_3 = generate_random_triangle_point()
        A = (x_1, y_1)
        B = (x_2, y_2)
        C = (x_3, y_3)
        point_list = np.array([A, B, C])
        area = cv2.contourArea(point_list)
    return cv2.drawContours(image_, [point_list], 0, color=color, thickness=-1), ((100 * area) / 9216) / 100

def draw_random_ellipse(image: Image, 
                        color: Tuple) -> np.ndarray:
    """
    Function used to randomly draw an ellipse on a black image.
    :param image: Blank image to draw the ellipse on.
    :param color: Tuple representing the image code in RGB format.
    :return: np.ndarray representing the image after drawing.
    """
    image_ = image.copy()
    x_1 = y_1 = 0
    x_1 = random.randint(36, 59)
    y_1 = random.randint(36, 59)
    major_axis = random.randint(10, 36)
    minor_axis = random.randint(10, 36)
    return cv2.ellipse(image_, (x_1, y_1), (major_axis, minor_axis), ANGLE, START_ANGLE, END_ANGLE, color=color, thickness=-1), ((100 * (3.142 * major_axis * minor_axis)) / 9216) / 100

def generate_dataset(num_of_samples: int = 9) -> None:
    """
    Function used to randomly generate input data if it's missing.
    :param num_of_samples: Number of samples that we want to generate as input data.
    """
    blank = generate_blank()
    for sample in tqdm(range(num_of_samples)):
        rand_state = random.randint(0, 8)   # Choose random number between 0-8 (we have 3 shapes + 3 colors => 3x3 = 9 options)
        if rand_state == 0:
            rectangle_sample, _ = draw_random_rectangle(blank, RED)
            if random.uniform(0, 1) > 0.5:
                noise = np.random.uniform(0, 127, (96, 96, 3))
                rectangle_sample = rectangle_sample + noise
                rectangle_sample = rectangle_sample.astype(int)
            cv2.imwrite(f"./input/{sample}.png", rectangle_sample)
        elif rand_state == 1:
            rectangle_sample, _ = draw_random_rectangle(blank, GREEN)
            if random.uniform(0, 1) > 0.5:
                noise = np.random.uniform(0, 127, (96, 96, 3))
                rectangle_sample = rectangle_sample + noise
                rectangle_sample = rectangle_sample.astype(int)
            cv2.imwrite(f"./input/{sample}.png", rectangle_sample)
        elif rand_state == 2:
            rectangle_sample, _ = draw_random_rectangle(blank, BLUE)
            if random.uniform(0, 1) > 0.5:
                noise = np.random.uniform(0, 127, (96, 96, 3))
                rectangle_sample = rectangle_sample + noise
                rectangle_sample = rectangle_sample.astype(int)
            cv2.imwrite(f"./input/{sample}.png", rectangle_sample)
        elif rand_state == 3:
            ellipse_sample, _ = draw_random_ellipse(blank, RED)
            if random.uniform(0, 1) > 0.5:
                noise = np.random.uniform(0, 127, (96, 96, 3))
                ellipse_sample = ellipse_sample + noise
                ellipse_sample = ellipse_sample.astype(int)
            cv2.imwrite(f"./input/{sample}.png", ellipse_sample)
        elif rand_state == 4:
            ellipse_sample, _ = draw_random_ellipse(blank, GREEN)
            if random.uniform(0, 1) > 0.5:
                noise = np.random.uniform(0, 127, (96, 96, 3))
                ellipse_sample = ellipse_sample + noise
                ellipse_sample = ellipse_sample.astype(int)
            cv2.imwrite(f"./input/{sample}.png", ellipse_sample)
        elif rand_state == 5:
            ellipse_sample, _ = draw_random_ellipse(blank, BLUE)
            if random.uniform(0, 1) > 0.5:
                noise = np.random.uniform(0, 127, (96, 96, 3))
                ellipse_sample = ellipse_sample + noise
                ellipse_sample = ellipse_sample.astype(int)
            cv2.imwrite(f"./input/{sample}.png", ellipse_sample)
        elif rand_state == 6:
            triangle_sample, _ = draw_random_triangle(blank, RED)
            if random.uniform(0, 1) > 0.5:
                noise = np.random.uniform(0, 127, (96, 96, 3))
                triangle_sample = triangle_sample + noise
                triangle_sample = triangle_sample.astype(int)
            cv2.imwrite(f"./input/{sample}.png", triangle_sample)
        elif rand_state == 7:
            triangle_sample, _ = draw_random_triangle(blank, GREEN)
            if random.uniform(0, 1) > 0.5:
                noise = np.random.uniform(0, 127, (96, 96, 3))
                triangle_sample = triangle_sample + noise
                triangle_sample = triangle_sample.astype(int)
            cv2.imwrite(f"./input/{sample}.png", triangle_sample)
        elif rand_state == 8:
            triangle_sample, _ = draw_random_triangle(blank, BLUE)
            if random.uniform(0, 1) > 0.5:
                noise = np.random.uniform(0, 127, (96, 96, 3))
                triangle_sample = triangle_sample + noise
                triangle_sample = triangle_sample.astype(int)
            cv2.imwrite(f"./input/{sample}.png", triangle_sample)

def compute_color_and_area(image: Image) -> Tuple[str, float]:
    """
    Function used to compute the color and area in percentages of a shape in an image.
    :param image: Source image to compute color and area occupied in percentages.
    :return: The result is a tuple made of a color and an area measured in %.
    """
    image_ = image.copy()
    image_ = np.array(image_)
    image_ = cv2.cvtColor(image_, cv2.COLOR_RGB2BGR)
    image_[image_ < (127,127,127)] = 0

    # Grayscale
    gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200)
    
    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0 :
        areas = [cv2.contourArea(cnt) for cnt in contours]
        max_area_idx = np.argmax(areas)
        x, y = contours[max_area_idx].T
        left, top, right, bottom = (np.min(x), np.min(y), np.max(x), np.max(y))

        # Draw all contours
        # -1 signifies drawing all contours
        area = cv2.contourArea(contours[max_area_idx])
        image = np.array(image)
        center = [int((left+right)/2), int((bottom+top)/2)]
        r, g, b = image[center[1]][center[0]]
        if r > 127 :
            return "red", round(((100 * area) / 9216) / 100, 3)
        elif g > 127 :
            return "green", round(((100 * area) / 9216) / 100, 3)
        elif b > 127 :
            return "blue", round(((100 * area) / 9216) / 100, 3)
        return "None", 0.000
    
    else :
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # aici
        image_ = np.array(image.copy())
        image_ = cv2.cvtColor(image_, cv2.COLOR_RGB2BGR)
        image_[image_ < (127,127,127)] = 0

        # Grayscale
        gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)

        # Find Canny edges
        edged = cv2.Canny(gray, 30, 200)

        # Finding Contours
        # Use a copy of the image e.g. edged.copy()
        # since findContours alters the image
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        areas = [cv2.contourArea(cnt) for cnt in contours]
        max_area_idx = np.argmax(areas)
        x, y = contours[max_area_idx].T
        left, top, right, bottom = (np.min(x), np.min(y), np.max(x), np.max(y))

        # Draw all contours
        # -1 signifies drawing all contours
        area = cv2.contourArea(contours[max_area_idx])
        #print(f'aria formei: {((100 * area) / 9216) / 100}')
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB) # aici
        image = np.array(image)
        center = [int((left+right)/2), int((bottom+top)/2)]
        r, g, b = image[center[1]][center[0]]
        if r > 127 :
            return "red", round(((100 * area) / 9216) / 100, 3)
        elif g > 127 :
            return "green", round(((100 * area) / 9216) / 100, 3)
        elif b > 127 :
            return "blue", round(((100 * area) / 9216) / 100, 3)
        return "None", 0.000

if __name__ == '__main__':

    resnet18_shape = torchvision.models.resnet18(pretrained=False)

    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(512, 128)),
        ('dropout', nn.Dropout(p=.5)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(128, 3)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    resnet18_shape.fc = fc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18_shape.to(device)

    # Load net used for shape detection
    resnet18_shape.load_state_dict(torch.load(r"./resnet18_shape.pth"))
    resnet18_shape.eval()

    if not os.path.isdir("./input"):
        os.mkdir("./input")
        generate_dataset(num_of_samples = NUM_OF_SAMPLES)
        print("Input folder was missing. Successfully generated some input data.")

    else :
        if len(os.listdir("./input")) == 0:
            generate_dataset(num_of_samples = NUM_OF_SAMPLES)
            print("Input folder was empty. Successfully generated some input data.")
        
    size = 0
    while size**2 < len(os.listdir("./input")):
        size = size + 1
    size = size**2

    if len(os.listdir("./input")) > 9:
        print("Due to the fact that the number of samples is >9, the display may look uglier. We suggest using <= 9 samples as input.")

    files = list([pth for pth in os.listdir("./input")])
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(top = .9, bottom=.05, hspace=.9, wspace=.05)

    for idx, filename in enumerate(files):
        sample = Image.open(os.path.join("./input/",filename))

        sample_ = sample.copy()
        sample_ = np.array(sample_)

        sample_ = sample_ / sample_.max()
        sample_ = sample_.transpose(2, 0, 1)
        sample_ = torch.Tensor(sample_)
        sample_ = sample_.unsqueeze(0)

        shape = resnet18_shape(sample_.to(device))
        if str(torch.argmax(shape.cpu(), axis=1)) == 'tensor([0])':
            shape = "ellipse"
        elif str(torch.argmax(shape.cpu(), axis=1)) == 'tensor([1])':
            shape = "rectangle"
        elif str(torch.argmax(shape.cpu(), axis=1)) == 'tensor([2])':
            shape = "triangle"

        sample_ = sample.copy()
        color, area = compute_color_and_area(sample_)

        fig.add_subplot(int(np.sqrt(size)), int(np.sqrt(size)), idx+1)
        plt.imshow(sample)
        plt.title(f"{color} {shape} {area}%")
    
    plt.show()



            

            





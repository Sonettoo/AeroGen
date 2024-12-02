import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
max_length=10

def resize_image_and_bboxes(image, bboxes, target_size=(512, 512)):
    """Resize image and bounding boxes to target size."""
    original_size = (800,800)
    image_resized = cv2.resize(image, target_size)
    scale_x = target_size[1] / original_size[1]
    scale_y = target_size[0] / original_size[0]
    
    resized_bboxes = []
    for bbox in bboxes:
        x_coords = np.array(bbox[0]) * scale_x
        y_coords = np.array(bbox[1]) * scale_y
        resized_bboxes.append((x_coords, y_coords))
    
    return image_resized, resized_bboxes

def random_rotation(image, bboxes, degrees):
    """Rotate the image and bounding boxes by a specific degree."""
    height, width = image.shape[:2]

    if degrees == 0:
        return image, bboxes

    M = cv2.getRotationMatrix2D((width // 2, height // 2), degrees, 1.0)
    cos_angle = np.abs(M[0, 0])
    sin_angle = np.abs(M[0, 1])
    new_width = int(height * sin_angle + width * cos_angle)
    new_height = int(height * cos_angle + width * sin_angle)
    M[0, 2] += (new_width / 2) - (width / 2)
    M[1, 2] += (new_height / 2) - (height / 2)

    image_rot = cv2.warpAffine(image, M, (new_width, new_height))
    
    new_bboxes = []
    for bbox in bboxes:
        points = np.array([[x, y] for x, y in zip(bbox[0], bbox[1])])
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])
        transformed_points = M.dot(points_ones.T).T
        new_x_coords = transformed_points[:, 0]
        new_y_coords = transformed_points[:, 1]
        new_bboxes.append((new_x_coords, new_y_coords))
    
    return image_rot, new_bboxes

def rotate_point(px, py, cx, cy, angle):
    """Rotate a point around a center by an angle."""
    rad = np.radians(angle)
    x_shifted = px - cx
    y_shifted = py - cy
    
    new_x = x_shifted * np.cos(rad) - y_shifted * np.sin(rad)
    new_y = x_shifted * np.sin(rad) + y_shifted * np.cos(rad)
    
    return new_x + cx, new_y + cy

def calculate_max_square(corners):
    """Calculate the largest square that fits within the given corners."""
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    square_size = int(min(x_max - x_min, y_max - y_min))
    square_center_x = int((x_min + x_max) // 2)
    square_center_y = int((y_min + y_max) // 2)
    
    x_min_square = max(0, square_center_x - square_size // 2)
    y_min_square = max(0, square_center_y - square_size // 2)
    x_max_square = x_min_square + square_size
    y_max_square = y_min_square + square_size

    return x_min_square, y_min_square, x_max_square, y_max_square

def crop_and_resize(image, corners, target_size=(512, 512)):
    """Crop the image to the largest square and resize to target size."""
    x_min_square, y_min_square, x_max_square, y_max_square = calculate_max_square(corners)

    image_cropped = image[int(y_min_square):int(y_max_square), int(x_min_square):int(x_max_square)]
    
    # Check if the cropped image is empty
    if image_cropped.size == 0:
        raise ValueError("Cropped image is empty. Check the bounding box coordinates.")
    
    image_resized = cv2.resize(image_cropped, target_size)

    return image_resized

class MyDataset(Dataset):
    def __init__(self, original_path, image_file_path, resolution, mask_size, mode='train', rotate_once=False, rotate_twice=False):
        self.original_path = original_path
        self.image_file_path = image_file_path
        self.resolution = (resolution, resolution)
        self.mask_size = (mask_size, mask_size)
        self.rotate_once = rotate_once
        self.rotate_twice = rotate_twice
        self.category = {
            "airplane": 0, "airport": 1, "baseballfield": 2, "basketballcourt": 3, "bridge": 4,
            "chimney": 5, "dam": 6, "Expressway-Service-area": 7, "Expressway-toll-station": 8,
            "golffield": 9, "groundtrackfield": 10, "harbor": 11, "overpass": 12, "ship": 13,
            "stadium": 14, "storagetank": 15, "tenniscourt": 16, "trainstation": 17, "vehicle": 18,
            "windmill": 19
        }
        self.descriptions = {
            "airplane": "airplane parked on the ground",
            "airport": "busy airport",
            "baseballfield": "green baseball field",
            "basketballcourt": "outdoor basketball court",
            "bridge": "long bridge",
            "chimney": "tall chimney",
            "dam": "large dam",
            "Expressway-Service-area": "crowded expressway service area",
            "Expressway-toll-station": "busy expressway toll station",
            "golffield": "well-maintained golf field",
            "groundtrackfield": "athletic ground track field",
            "harbor": "bustling harbor",
            "overpass": "elevated overpass",
            "ship": "ship on the water",
            "stadium": "large stadium",
            "storagetank": "industrial storage tank",
            "tenniscourt": "clay tennis court",
            "trainstation": "crowded train station",
            "vehicle": "vehicle on the road",
            "windmill": "rotating windmill"
        }
        self.category_embeddings = np.load('./datasets/category_embeddings.npy')
        self.condition_dropout_prob = 0.1
        self.mode = mode


        txt_file_path = f'./datasets/DIOR-VOC/VOC2007/ImageSets/Main/{mode}.txt'
        with open(txt_file_path, 'r') as file:
            file_names = file.read().splitlines()
            # print(file_names)
        
        self.files = [os.path.join(original_path, f'{file_name}.xml') for file_name in file_names]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        xml_file_path = self.files[idx]
        # print(xml_file_path)
        image, filename, img_size, bboxes, class_names, _ = self._parse_annotation(xml_file_path)
        filename=xml_file_path.split('/')[-1].replace('xml','jpg')
        prompt = self._generate_prompt(class_names)
        
        # 图像和bbox增强
        image_resized, bboxes_resized = resize_image_and_bboxes(image, bboxes, target_size=(512, 512))
        
        if self.rotate_once or self.rotate_twice:
            angle = random.choice([0, 90, 180, 270])
            # print(angle)
            rot_image, rot_bboxes = random_rotation(image_resized, bboxes_resized, angle)
            if self.rotate_twice:
                angle = random.uniform(-15, 15)
                rot_image, rot_bboxes = random_rotation(rot_image, rot_bboxes, angle)
        else:
            rot_image, rot_bboxes = image_resized, bboxes_resized
        if self.rotate_twice:
        # 初始化四个角的坐标
            height, width = rot_image.shape[:2]
            corners = np.array([[0, 0], [0, height], [width, 0], [width, height]])
            
            # 计算旋转后的角坐标
            corners_rotated = np.array([rotate_point(x, y, width // 2, height // 2, angle) for x, y in corners])
            
            if self.rotate_twice:
                # 裁剪并 resize
                rot_image = crop_and_resize(rot_image, corners_rotated, target_size=(512, 512))
        
        mask_conditions, rot_bboxes, mask_vector = self._generate_mask(rot_bboxes, self.resolution, self.mask_size)
        category_conditions = self._get_category_conditions(class_names)
        
        return dict(
            jpg=rot_image,
            filename=filename,
            txt=prompt,
            mask_conditions=mask_conditions,
            bboxes=rot_bboxes,
            category_conditions=category_conditions,
            mask_vector=mask_vector
        )

    def _generate_prompt(self, class_names):
        if self.mode == 'train':
            if random.random() < self.condition_dropout_prob:
                return ''

        unique_class_names = list(set(class_names))
        prompt = 'an aerial image with '
        if len(unique_class_names) == 1:
            prompt += self.descriptions[unique_class_names[0]]
        else:
            prompt += ', '.join([self.descriptions[name] for name in unique_class_names[:-1]]) + ' and ' + self.descriptions[unique_class_names[-1]]
        return prompt

    def _generate_mask(self, bboxes, source_size, target_size):
        source_height, source_width = (512,512)  # after resize DIOR dataset
        target_height, target_width = target_size
        
        mask_list = []
        bbox_list = []
        mask_vector = np.zeros(max_length, dtype=np.float32)
        
        for i, bbox in enumerate(bboxes):
            if i >= max_length:
                break
            x_coords, y_coords = bbox
            x_coords_scaled = (np.array(x_coords) * target_width / source_width).astype(int)
            y_coords_scaled = (np.array(y_coords) * target_height / source_height).astype(int)
            
            mask = np.zeros((target_height, target_width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.stack((x_coords_scaled, y_coords_scaled), axis=-1)], 1)
            mask_list.append(mask)
            bbox_list.append(np.stack((x_coords_scaled, y_coords_scaled), axis=-1).flatten())
            mask_vector[i] = 1
        
        # Pad with empty masks and zero bboxes if needed
        while len(mask_list) < max_length:
            mask_list.append(np.zeros((target_height, target_width), dtype=np.uint8))
            bbox_list.append(np.zeros(8, dtype=int))
        
        return np.stack(mask_list), np.stack(bbox_list), mask_vector

    def _get_category_conditions(self, class_names):
        category_conditions = []
        for class_name in class_names:
            category_index = self.category[class_name]
            category_embedding = self.category_embeddings[category_index]
            category_conditions.append(category_embedding)
        
        # Limit the category_conditions length to max_length
        if len(category_conditions) > max_length:
            category_conditions = category_conditions[:max_length]
        
        # Pad with zeros if needed
        while len(category_conditions) < max_length:
            category_conditions.append(np.zeros(768))
        
        return np.stack(category_conditions)

    def _parse_annotation(self, xml_file_path):
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        filename = root.find('filename').text
        # print(filename)
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)
        img_size = (width, height, depth)
        
        objects = root.findall('object')
        bboxes = []
        class_names = []
        prompt_parts = []
        
        for obj in objects:
            name = obj.find('name').text
            class_names.append(name)

            robndbox = obj.find('robndbox')
            x_left_top = int(robndbox.find('x_left_top').text)
            y_left_top = int(robndbox.find('y_left_top').text)
            x_right_top = int(robndbox.find('x_right_top').text)
            y_right_top = int(robndbox.find('y_right_top').text)
            x_right_bottom = int(robndbox.find('x_right_bottom').text)
            y_right_bottom = int(robndbox.find('y_right_bottom').text)
            x_left_bottom = int(robndbox.find('x_left_bottom').text)
            y_left_bottom = int(robndbox.find('y_left_bottom').text)
            
            x_coords = [x_left_top, x_right_top, x_right_bottom, x_left_bottom]
            y_coords = [y_left_top, y_right_top, y_right_bottom, y_left_bottom]
            bboxes.append((x_coords, y_coords))
            
            prompt_parts.append(f"{name} at [{x_left_top}, {y_left_top}, {x_right_top}, {y_right_top}, {x_right_bottom}, {y_right_bottom}, {x_left_bottom}, {y_left_bottom}]")
        
        prompt = " and ".join(prompt_parts)
        
        # image_path = os.path.join(self.image_file_path, filename)
        
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, self.resolution)
        # image = (image.astype(np.float32) / 127.5) - 1.0
        image = np.ones((self.resolution[1], self.resolution[0], 3), dtype=np.float32) * 255
        
        return image, filename, img_size, bboxes, class_names, prompt

if __name__ == "__main__":
    
    original_path = ''
    image_file_path = ''
    npy_file_path = ''
    resolution = 512
    mask_size = 64

    # 设置是否进行一次旋转或两次旋转
    dataset = MyDataset(original_path, image_file_path, resolution, mask_size, rotate_once=False, rotate_twice=False)

    # 获取数据并打印
    for idx in range(len(dataset)):
        data = dataset[idx]
        print(data['filename'])

    # dataset = MyDataset(original_path, image_file_path, resolution, mask_size, mode='val')
    # from torch.utils.data import DataLoader, Dataset
    # # 创建DataLoader，启用多线程
    # dataloader = DataLoader(dataset, batch_size=8, num_workers=64)

    # # 获取数据并打印
    # from tqdm import tqdm
    # for data in tqdm(dataloader):
    #     # print(data['filename'])
    #     a=data

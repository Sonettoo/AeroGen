import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

class MyDataset(Dataset):
    def __init__(self, original_path, resolution, mask_size, mode='test', rotate_aug=False, max_length=10):
        self.original_path = original_path
        self.resolution = (resolution, resolution)
        self.mask_size = (mask_size, mask_size)
        self.condition_dropout_prob = 0.1
        self.category_mapping = {
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
            "harbor": "bustling harbour by the sea",
            "overpass": "elevated overpass",
            "ship": "ship on the water",
            "stadium": "large stadium",
            "storagetank": "industrial storage tank",
            "tenniscourt": "clay tennis court",
            "trainstation": "crowded train station",
            "vehicle": "vehicle on the road",
            "windmill": "rotating windmill"
        }
        self.mode = mode
        if mode != 'train':
            self.max_length = 20
        else:
            self.max_length = 20
        self.category_embeddings = np.load('./datasets/category_embeddings.npy')
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"The directory {original_path} does not exist. Please check the path.")

        self.files = [os.path.join(original_path, f) for f in os.listdir(original_path) if f.endswith('.txt')]

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .txt files found in the directory {original_path}. Please check if the directory contains valid .txt files.")

        self.rotate_aug = rotate_aug

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        txt_file_path = self.files[idx]
        filename = os.path.basename(txt_file_path)
        bboxes, class_names = self._parse_annotation(txt_file_path)
        
        img_size = (512, 512)
        scale_x = 512 / img_size[0]
        scale_y = 512 / img_size[1]

        bboxes = np.array(bboxes) * [scale_x, scale_y, scale_x, scale_y, scale_x, scale_y, scale_x, scale_y]

        
        prompt = self._generate_prompt(class_names)

        # Perform random rotation
        rotation_angle = random.choice([0, 90, 180, 270])
        if rotation_angle != 0 and self.rotate_aug and self.mode == 'train':
            bboxes = self._random_rotation(bboxes, rotation_angle)

        mask_conditions, bboxes, mask_vector = self._generate_mask(bboxes, (512, 512), self.mask_size)
        category_conditions, category_labels = self._get_category_conditions(class_names)

        return {
            'filename': filename,
            'txt': prompt,
            'mask_conditions': mask_conditions,
            'bboxes': bboxes,
            'category_conditions': category_conditions,
            'category_labels': category_labels,
            'mask_vector': mask_vector
        }

    def _parse_annotation(self, txt_file_path):
        """
        """
        bboxes = []
        class_names = []
        
        # 读取txt文件
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            values = line.strip().split()
            class_id = int(values[0])
            
            class_name = list(self.category_mapping.keys())[list(self.category_mapping.values()).index(class_id)]
            class_names.append(class_name)
            x1 = int(float(values[1]) * 512)
            y1 = int(float(values[2]) * 512)
            x2 = int(float(values[3]) * 512)
            y2 = int(float(values[4]) * 512)
            x3 = int(float(values[5]) * 512)
            y3 = int(float(values[6]) * 512)
            x4 = int(float(values[7]) * 512)
            y4 = int(float(values[8]) * 512)
            
            bboxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        
        return bboxes, class_names



    def _generate_prompt(self, class_names):
        if self.mode == 'train' and random.random() < self.condition_dropout_prob:
            return ''

        unique_class_names = list(set(class_names))
        prompt = 'an aerial image with '
        if len(unique_class_names) == 1:
            prompt += self.descriptions[unique_class_names[0]]
        else:
            prompt += ', '.join([self.descriptions[name] for name in unique_class_names[:-1]]) + ' and ' + self.descriptions[unique_class_names[-1]]
        return prompt

    def _random_rotation(self, bboxes, degrees):
        """Rotate the bounding boxes by a specific degree."""
        if degrees == 0:
            return bboxes

        # Perform bounding box rotation (the image rotation part has been removed)
        new_bboxes = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            points = np.array([
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax]
            ])
            # Perform rotation (placeholder as image data is not needed)
            transformed_points = points  # Here you would implement the actual rotation logic
            new_x_coords = transformed_points[:, 0]
            new_y_coords = transformed_points[:, 1]
            new_bboxes.append((min(new_x_coords), min(new_y_coords), max(new_x_coords), max(new_y_coords)))

        return np.array(new_bboxes)

    

    def _generate_mask(self, bboxes, source_size, target_size):
        source_height, source_width = source_size
        target_height, target_width = target_size
        
        mask_list = []
        bbox_list = []
        mask_vector = np.zeros(self.max_length, dtype=np.float32)
        
        for i, bbox in enumerate(bboxes):
            if i >= self.max_length:
                break
            
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox

            x1_scaled = int(x1 * target_width / source_width)
            y1_scaled = int(y1 * target_height / source_height)
            x2_scaled = int(x2 * target_width / source_width)
            y2_scaled = int(y2 * target_height / source_height)
            x3_scaled = int(x3 * target_width / source_width)
            y3_scaled = int(y3 * target_height / source_height)
            x4_scaled = int(x4 * target_width / source_width)
            y4_scaled = int(y4 * target_height / source_height)
            
            mask = Image.new('L', (target_width, target_height), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (x3_scaled, y3_scaled), (x4_scaled, y4_scaled)], outline=1, fill=1)
            
            mask_list.append(np.array(mask))
            
            bbox_list.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled, x3_scaled, y3_scaled, x4_scaled, y4_scaled])
            mask_vector[i] = 1
        
        while len(mask_list) < self.max_length:
            mask_list.append(np.zeros((target_height, target_width), dtype=np.uint8))
            bbox_list.append([0, 0, 0, 0, 0, 0, 0, 0])
        
        return np.stack(mask_list), np.stack(bbox_list), mask_vector


    def _get_category_conditions(self, class_names):
        category_conditions = []
        labels = []

        for class_name in class_names:
            if class_name not in self.category_mapping:
                raise KeyError(f"Class name '{class_name}' not found in category_mapping.")
            
            category_index = self.category_mapping[class_name]
            category_embedding = self.category_embeddings[category_index]
            category_conditions.append(category_embedding)
            labels.append(category_index)

        if len(category_conditions) > self.max_length:
            category_conditions = category_conditions[:self.max_length]
            labels = labels[:self.max_length]

        while len(category_conditions) < self.max_length:
            category_conditions.append(np.zeros(768))
            labels.append(-1)

        return np.stack(category_conditions), np.array(labels)


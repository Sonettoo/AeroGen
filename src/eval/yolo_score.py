import os
import numpy as np
import json
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def read_file_indices(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def plot_results(image_path, boxes, save_path):
    image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for box in boxes:
        xmin, ymin, xmax, ymax, score, class_id = box
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=2))
        ax.text(xmin, ymin, f'{score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def save_inference_results(image_folder, results, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for result in results:
        image_path = result.path
        file_name = os.path.basename(image_path)
        save_path = os.path.join(output_folder, file_name)
        boxes = [(*bbox[:4].tolist(), bbox[4].item(), int(bbox[5])) for bbox in result.boxes.data]
        plot_results(image_path, boxes, save_path)

def xml_to_coco(xml_folder, image_folder, txt_file, xml_category, filename_to_id):
    scale = 0.64
    file_indices = read_file_indices(txt_file)
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    annotation_id = 1
    for file_index in file_indices:
        xml_file = f"{file_index}.xml"
        if os.path.exists(os.path.join(xml_folder, xml_file)):
            tree = ET.parse(os.path.join(xml_folder, xml_file))
            root = tree.getroot()
            filename = root.find('filename').text
            image_id = filename_to_id[filename]
            
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            coco_data['images'].append({
                "file_name": filename,
                "height": height,
                "width": width,
                "id": image_id
            })
            
            for obj in root.findall('object'):
                category = obj.find('name').text
                category_id = xml_category[category]
                bndbox = obj.find('bndbox')
                xmin = int(int(bndbox.find('xmin').text) * scale)
                ymin = int(int(bndbox.find('ymin').text) * scale)
                xmax = int(int(bndbox.find('xmax').text) * scale)
                ymax = int(int(bndbox.find('ymax').text) * scale)
                coco_data['annotations'].append({
                    "image_id": image_id,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "category_id": category_id + 1,  # COCO categories start at 1
                    "id": annotation_id,
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0
                })
                annotation_id += 1
                
    coco_data['categories'] = [{"id": v + 1, "name": k} for k, v in xml_category.items()]
    return coco_data

def yolo_to_coco_annotations(yolo_results, image_folder, filename_to_id):
    annotations = []
    annotation_id = 1

    for result in yolo_results:
        file_name = os.path.basename(result.path)
        if file_name not in filename_to_id:
            continue  # Skip images not in the mapping
        image_id = filename_to_id[file_name]
        # image_path = os.path.join(image_folder, file_name)
        # image = Image.open(image_path)
        # width, height = image.size

        for bbox in result.boxes.data:
            category_id = int(bbox[5])
            xmin, ymin, xmax, ymax = bbox[:4].tolist()
            annotations.append({
                "image_id": image_id,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "category_id": category_id + 1,  # COCO categories start at 1
                "id": annotation_id,
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0,
                "score": bbox[4].item()
            })
            annotation_id += 1

    return annotations

def main(xml_folder, image_folder, output_folder, txt_file, batch_size):
    model = YOLO('best.pt')
    
    #----------------------------------delete
    # class_names = model.names
    # xml_category = {
    #     "airplane": 0, "airport": 1, "baseballfield": 2, "basketballcourt": 3, "bridge": 4, 
    #     "chimney": 5, "dam": 6, "Expressway-Service-area": 7, "Expressway-toll-station": 8, 
    #     "golffield": 9, "groundtrackfield": 10, "harbor": 11, "overpass": 12, "ship": 13, 
    #     "stadium": 14, "storagetank": 15, "tenniscourt": 16, "trainstation": 17, "vehicle": 18, 
    #     "windmill": 19
    # }
    # print(class_names)
    #----------------------------------

    #----------------------------------add
    class_names = model.names
    print("Model class names:", class_names)

    xml_category = {name: id for id, name in class_names.items()}
    print("XML Category Mapping:", xml_category)
    #----------------------------------

    
    file_indices = read_file_indices(txt_file)
    filename_to_id = {f"{file_index}.jpg": idx for idx, file_index in enumerate(file_indices)}
    
    image_paths = [os.path.join(image_folder, f"{file_index}.jpg") for file_index in file_indices if os.path.exists(os.path.join(image_folder, f"{file_index}.jpg"))]
    
    all_results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        results = model(batch_paths)
        all_results.extend(results)
    
    save_inference_results(image_folder, all_results, output_folder)
    yolo_annotations = yolo_to_coco_annotations(all_results, image_folder, filename_to_id)
    with open('yolo_result.json', 'w') as f:
        json.dump(yolo_annotations, f, cls=MyEncoder)
    
    xml_coco_data = xml_to_coco(xml_folder, image_folder, txt_file, xml_category, filename_to_id)
    with open('xml_result.json', 'w') as f:
        json.dump(xml_coco_data, f, cls=MyEncoder)
    
    coco_gt = COCO('xml_result.json')
    coco_dt = coco_gt.loadRes('yolo_result.json')
    gt_image_ids = set(coco_gt.getImgIds())
    dt_image_ids = set(coco_dt.getImgIds())

    common_image_ids = list(gt_image_ids & dt_image_ids)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = common_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print(f"mAP: {coco_eval.stats[0]}")
    print(f"mAP50: {coco_eval.stats[1]}")
    print(f"mAP75: {coco_eval.stats[2]}")

xml_folder = "./dataset/DIOR-VOC/Annotations/Horizontal_Bounding_Boxes/"
image_folder = 'path/generation/image'
output_folder = './Yolov8_DIOR/pred'
txt_file = "./Main/test.txt"

main(xml_folder, image_folder, output_folder, txt_file, 128)

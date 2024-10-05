from ultralytics import YOLO
import cv2
import numpy as np

def process_image_with_yolo(model_path, image_path):
    def load_model(model_path):
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

    def perform_inference(model, image_path):
        try:
            results = model(image_path)
            return results
        except Exception as e:
            print(f"Error performing inference: {e}")
            exit(1)

    def load_image(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    
        if image is None:
            print(f"Error loading image: {image_path}")
            exit(1)
        return image

    def preprocess_image(image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_enhanced = clahe.apply(image)
        blurred = cv2.GaussianBlur(image_enhanced, (9, 9), 10.0)
        sharpened = cv2.addWeighted(image_enhanced, 1.5, blurred, -0.5, 0)
        return sharpened

    def iou(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)

        intersection = max(0, x_max - x_min) * max(0, y_max - y_min)

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def draw_bounding_boxes(image, boxes_to_draw, class_names, class_colors):
        for box_data in boxes_to_draw:
            x_min, y_min, x_max, y_max = box_data["coords"]
            cls = box_data["cls"]
            conf = box_data["conf"]

            class_name = class_names.get(cls, f"Class {cls}")
            label = f'{class_name}: {conf:.2f}'

            color = class_colors.get(cls, (0, 0, 0))  # Default to black if color not defined

            overlay = image.copy()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
            alpha = 0.4
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            label_y = max(y_min, 20)

            cv2.putText(image, label, (x_min, label_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 250, 0), 2, lineType=cv2.LINE_AA)

        return image

    # Load model
    model = load_model(model_path)

    # Perform inference
    results = perform_inference(model, image_path)

    # Load and preprocess image
    image = load_image(image_path)
    image_processed = preprocess_image(image)
    image_bgr = cv2.cvtColor(image_processed, cv2.COLOR_GRAY2BGR)

    class_names = {
        0: "Aortic_enlargement",
        1: "Atelectasis",
        2: "Calcification",
        3: "Cardiomegaly",
        4: "Consolidation",
        5: "ILD",
        6: "Infiltration",
        7: "Lung_Opacity",
        8: "Nodule/Mass",
        9: "Other_lesion",
        10: "Pleural_effusion",
        11: "Pleural_thickening",
        12: "Pneumothorax",
        13: "Pulmonary_fibrosis",
        14: "No finding"
    }

    class_colors = {
        0: (0, 255, 0),    # Green for Aortic_enlargement
        1: (255, 0, 0),    # Red for Atelectasis
        2: (0, 0, 255),    # Blue for Calcification
        3: (255, 255, 0),  # Cyan for Cardiomegaly
        4: (0, 255, 255),  # Yellow for Consolidation
        5: (255, 0, 255),  # Magenta for ILD
        6: (128, 0, 128),  # Purple for Infiltration
        7: (255, 165, 0),  # Orange for Lung_Opacity
        8: (0, 128, 128),  # Teal for Nodule/Mass
        9: (128, 128, 0),  # Olive for Other_lesion
        10: (0, 0, 128),   # Navy for Pleural_effusion
        11: (128, 128, 128),# Gray for Pleural_thickening
        12: (255, 192, 203),# Pink for Pneumothorax
        13: (0, 255, 127),  # Spring Green for Pulmonary_fibrosis
        14: (255, 255, 255), # Light Gray for No finding (or any other color)
    }

    boxes_to_draw = []

    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf.cpu().numpy().item())
            cls = int(box.cls.cpu().numpy().item())

            if conf > 0.1:
                overlapping = False
                for drawn_box in boxes_to_draw:
                    if iou(drawn_box["coords"], (x_min, y_min, x_max, y_max)) > 0.5:
                        overlapping = True
                        if conf > drawn_box["conf"]:
                            drawn_box.update({"coords": (x_min, y_min, x_max, y_max), "cls": cls, "conf": conf})
                        break
                
                if not overlapping:
                    boxes_to_draw.append({"coords": (x_min, y_min, x_max, y_max), "cls": cls, "conf": conf})

    # Draw bounding boxes
    output_image = draw_bounding_boxes(image_bgr, boxes_to_draw, class_names, class_colors)

    # Extract classes and confidence scores
    detected_classes = [(class_names[box["cls"]], box["conf"]) for box in boxes_to_draw]

    return output_image, detected_classes

# Example usage
if __name__ == '__main__':
    model_path = 'Yolov8l(640, 50, 16)/weights/best.pt'
    image_path = 'image4.jpeg'
    
    output_image, detected_classes = process_image_with_yolo(model_path, image_path)
    cv2.imwrite('output_image.jpg', output_image)
    
    # Print detected classes and their confidence scores
    for class_name, confidence in detected_classes:
        print(f'Detected: {class_name} with confidence: {confidence:.2f}')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
warnings.filterwarnings("ignore")
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Yolo(nn.Module):
    def __init__(self, grid_size, num_boxes, num_classes):
        super(Yolo, self).__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # conv1_1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # conv1_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),       # 224->112

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # conv2_1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# conv2_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),        # 112->56

            nn.Conv2d(128, 256, kernel_size=3, padding=1),# conv3_1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# conv3_2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# conv3_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),        # 56->28

            nn.Conv2d(256, 512, kernel_size=3, padding=1),# conv4_1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# conv4_2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# conv4_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),        # 28->14

            nn.Conv2d(512, 512, kernel_size=3, padding=1),# conv5_1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# conv5_2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# conv5_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),        # 14->7
        )
        self.detector = nn.Sequential(
            nn.Linear(512 * self.S * self.S, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.detector(x)
        x = F.sigmoid(x)
        x = x.view(-1, self.S, self.S, self.B*5+self.C)
        return x


def NMS(bboxes, scores, threshold=0.35):
    ''' Non Max Suppression
    Args:
        bboxes: (torch.tensors) list of bounding boxes. size:(N, 4) ((left_top_x, left_top_y, right_bottom_x, right_bottom_y), (...))
        probs: (torch.tensors) list of confidence probability. size:(N,)
        threshold: (float)
    Returns:
        keep_dim: (torch.tensors)
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        try:
            i = order[0]
        except:
            i = order.item()
        keep.append(i)

        if order.numel() == 1: break

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    keep_dim = torch.LongTensor(keep)
    return keep_dim

def decoder(grid, num_boxes, num_classes):
    """ Decoder function that decode the output-grid to bounding box, class and probability.
    Args:
        grid: (torch.tensors) output grid from YOLO. size: (1, S, S, B*5 + 20)
    Returns:
        bboxes: (torch.tensors) list of bounding boxes. size:(N, 4) ((left_top_x, left_top_y, right_bottom_x, right_bottom_y), (...))
        class_idxs: (torch.tensors) list of class index. size:(N,)
        probs: (torch.tensors) list of confidence probability. size:(N,)
    """

    grid_num = 7
    bboxes = []
    class_idxs = []
    probs = []

    grid = grid.squeeze()
    assert grid.size() == (grid_num, grid_num, num_boxes * 5 + num_classes)
    S, B, C = grid_num, num_boxes, num_classes

    for i in range(S):
        for j in range(S):
            cell = grid[i, j]
            class_probs = cell[B*5:]  # [20]

            for b in range(B):
                box = cell[b*5:(b+1)*5]
                conf = box[4]
                if conf < 0.2:  # confidence threshold
                    continue

                # bbox center x,y in absolute coord (0~S)
                cx = (box[0] + j) / S
                cy = (box[1] + i) / S
                w = box[2]
                h = box[3]

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                class_prob, class_idx = torch.max(class_probs, dim=0)
                prob = conf * class_prob

                if prob < 0.2:  # final probability threshold
                    continue

                bboxes.append(torch.tensor([x1, y1, x2, y2]).unsqueeze(0))
                class_idxs.append(torch.tensor([class_idx]))
                probs.append(torch.tensor([prob]))

    if len(bboxes) == 0: # Any box was not detected
        bboxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        class_idxs = torch.zeros(1, dtype=torch.int)

    else:
        #list of tensors -> tensors
        bboxes = torch.cat(bboxes, dim=0)
        probs = torch.cat(probs, dim=0)
        class_idxs = torch.cat(class_idxs, dim=0)

    bboxes_result, class_idxs_result, probs_result = [], [], []
    for label in range(num_classes):
        label_mask = (class_idxs==label)
        if label_mask.sum() > 0:
            _bboxes = bboxes[label_mask]
            _probs = probs[label_mask]
            _class_idxs = class_idxs[label_mask]

            keep_dim = NMS(_bboxes, _probs, threshold=0.16) # Non Max Suppression
            bboxes_result.append(_bboxes[keep_dim])
            class_idxs_result.append(_class_idxs[keep_dim])
            probs_result.append(_probs[keep_dim])

    bboxes_result = torch.cat(bboxes_result, 0)
    class_idxs_result = torch.cat(class_idxs_result, 0)
    probs_result = torch.cat(probs_result, 0)

    return bboxes_result, class_idxs_result, probs_result


def inference_image(model, image, device, VOC_CLASSES, num_boxes=2, num_classes=20):
    original_image = image.copy()
    h, w, c = original_image.shape
    img = cv2.resize(original_image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = transform(torch.from_numpy(img).float().div(255).transpose(2, 1).transpose(1, 0))
    img = img.unsqueeze(0).to(device)

    output_grid = model(img).cpu()
    bboxes, class_idxs, probs = decoder(output_grid, num_boxes, num_classes)
    
    for i in range(bboxes.size(0)):
        bbox = bboxes[i]
        class_name = VOC_CLASSES[class_idxs[i]]
        prob = probs[i]
        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_image, f'{class_name}: {prob:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    return original_image


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.get_logger().info('YOLOv1 Detector Node is starting...')

        self.bridge = CvBridge()
        self.image_counter = 0

        self.input_dir = '~/ros2_ws/src/yolo_detector/img/input'
        self.output_dir = '~/ros2_ws/src/yolo_detector/img/output'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")

        self.VOC_CLASSES = (
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        )
        
        script_dir = '/home/ubuntu/ros2_ws/src/yolo_detector'
        ckpt_path =  '/home/ubuntu/ros2_ws/src/yolo_detector/checkpoints/best.pth'

        self.model = Yolo(grid_size=7, num_boxes=2, num_classes=20).to(self.device)
        
        self.get_logger().info(f"chkpt_path: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path))
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.get_logger().info('YOLOv1 model loaded successfully.')

        # Camera image Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/depth_cam/rgb/image_raw',  # Camera topic name
            self.image_callback,
            1)
        # self.get_logger().info("Camera Subscription Success")
	
    def image_callback(self, msg):
        self.get_logger().info(f'Received image frame {self.image_counter}')
        try:
            # ROS image to opencv img
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # 1. Save input image
        input_filename = os.path.join(self.input_dir, f"input_frame.jpg")
        self.image_counter += 1 
        cv2.imwrite(str(input_filename), cv_image)

        # 2. Inference image
        result_image = inference_image(
            self.model, 
            cv_image, 
            self.device, 
            self.VOC_CLASSES
        )
        cv2.imshow('YOLOv1 Realtime', result_image)
        # 3. Save output image
        output_filename = os.path.join(self.output_dir, f"result_frame.jpg")
        cv2.imwrite(str(output_filename), result_image)
        self.get_logger().info(f"Image No.{self.image_counter} processing complete.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Quit requested (q). Shutting down...')
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()

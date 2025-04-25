import cv2
import numpy as np
import onnxruntime as ort
from torchvision.ops import nms
import torch


def letterbox(inputImg, target_size=(640, 640)):
    h0, w0 = inputImg.shape[:2]
    r = min(target_size[0] / h0, target_size[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    img_resized = cv2.resize(inputImg, new_unpad)
    dw, dh = target_size[1] - new_unpad[0], target_size[0] - new_unpad[1]
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_padded, r, (dw, dh)


# 类别名称列表
class_names = ["CoverageTrap", "OpenedWindow", "OpenedDoor",
               "BrokeDoor", "ExternalSuspension", "OverflowGoods"]

# 加载模型
session = ort.InferenceSession(r"D:\Projects\model\best.onnx")
input_name = session.get_inputs()[0].name

# 读取与预处理
orig = cv2.imread("./ultralytics/datasets/Train/train/images/01001.jpg")
img, scale, (pad_w, pad_h) = letterbox(orig, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
img = img.transpose(2, 0, 1)[None, ...]

# 推理
outputs = session.run(None, {input_name: img})
preds = np.squeeze(outputs[0], axis=0)

# 提取
boxes = preds[:, :4]
scores = preds[:, 4:]
class_ids = np.argmax(scores, axis=1)
confidences = scores[np.arange(len(scores)), class_ids]

# 坐标反变换
x1 = (boxes[:, 0] - pad_w / 2) / scale
y1 = (boxes[:, 1] - pad_h / 2) / scale
x2 = (boxes[:, 2] - pad_w / 2) / scale
y2 = (boxes[:, 3] - pad_h / 2) / scale

# NMS
keep = nms(torch.tensor(np.stack([x1, y1, x2, y2], 1)),
           torch.tensor(confidences), iou_threshold=0.45).numpy()

# 绘制结果
for i in keep:
    cv2.rectangle(orig, (int(x1[i]), int(y1[i])),
                  (int(x2[i]), int(y2[i])), (0, 255, 0), 2)
    # label = f"{class_names[class_ids[i]]}:{confidences[i]:.2f}"
    # cv2.putText(orig, label, (int(x1[i]), int(y1[i]) - 5),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# 设置显示图像的大小
scale_percent = 20  # percent of original size
width = int(orig.shape[1] * scale_percent / 100)
height = int(orig.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
orig = cv2.resize(orig, dim, interpolation=cv2.INTER_AREA)
# 显示图像
cv2.imshow("Result", orig)
cv2.waitKey(0)

import os
import cv2
import torch
from models.YOLOX.tools.demo import Predictor, image_demo
from models.YOLOX.yolox.exp import get_exp
from models.YOLOX.yolox.data.datasets import COCO_CLASSES

def yolox_model(ckpt_file = 'features/models/YOLOX/weights/yolox_nano.pth',
                exp = 'yolox-nano',
                fp16 = True):
    
    exp = get_exp(exp_file=None, exp_name=exp)

    model = exp.get_model()
    
    if torch.cuda.is_available():
        model.cuda()
        if fp16:
            model.half()  # to FP16

    model.eval()
    
    ckpt = torch.load(ckpt_file, map_location="cpu")
    
    model.load_state_dict(ckpt["model"])

    trt_file = None
    decoder = None

    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, 'gpu', True, False)

    return predictor

def webcam_human_tracking(cls_conf = 0.6):
    predictor = yolox_model()


    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            
            continue
        outputs, img_info = predictor.inference(image)
        output = outputs[0].cpu()
        ratio = img_info["ratio"]

        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        classification = cls.cpu().to(dtype = int).tolist()

        for idx in range(bboxes.shape[0]):
            score = scores[idx]
            if score < cls_conf:
                continue
            else:
                name = COCO_CLASSES[classification[idx]]            
        
                if name == 'person':
                    bbox = bboxes.cpu().to(dtype = int).tolist()[idx]
                    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                else:
                    continue

        cv2.imshow('im', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break


def main():
    webcam_human_tracking()
    pass

if __name__ == '__main__':
    main()
    
    pass
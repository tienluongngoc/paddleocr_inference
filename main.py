from paddleocr.tools.infer.predict_rec import TextRecognizer
import paddleocr.tools.infer.utility as utility
import cv2
import os

if __name__ == "__main__":
    
    args = utility.parse_args()
    text_recognizer = TextRecognizer(args)
    img_list = []
    for fn in os.listdir(args.image_dir):
        img_list.append(cv2.imread(os.path.join(args.image_dir, fn)))

    rec_res, _ = text_recognizer(img_list)
    print(rec_res)

from paddleocr.tools.infer.predict_rec import TextRecognizer
import cv2
import numpy as np
import math
import time
import traceback

import paddleocr.tools.infer.utility as utility
from paddleocr.ppocr.postprocess import build_post_process
from paddleocr.ppocr.utils.logging import get_logger
from paddleocr.ppocr.utils.utility import get_image_file_list, check_and_read_gif
logger = get_logger()

def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextRecognizer(args)
    valid_image_file_list = []
    img_list = []

    # warmup 2 times
    if args.warmup:
        img = np.random.uniform(0, 255, [32, 320, 3]).astype(np.uint8)
        for i in range(2):
            res = text_recognizer([img] * int(args.rec_batch_num))

    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        rec_res, _ = text_recognizer(img_list)

    except Exception as E:
        logger.info(traceback.format_exc())
        logger.info(E)
        exit()
    for ino in range(len(img_list)):
        logger.info("Predicts of {}:{}".format(valid_image_file_list[ino],
                                               rec_res[ino]))
    if args.benchmark:
        text_recognizer.autolog.report()


if __name__ == "__main__":
    main(utility.parse_args())

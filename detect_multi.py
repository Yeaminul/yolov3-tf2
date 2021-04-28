import time
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import cv2
import tensorflow as tf
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('pbLoad', './serving/yolov3/15092020_adam_custom_anchors', 'path to saved_model')
flags.DEFINE_string('output', './data/detections/', 'path to prediction result')
flags.DEFINE_string('classes', './data/waste.names', 'path to classes file')
flags.DEFINE_string('image_dir', './data/waste/JPEGImages/', 'directory to input image')
flags.DEFINE_string('data_dir', './data/waste/ImageSets/Main/', 'directory to train/test/val files data')
flags.DEFINE_enum('mode', 'val', ['train', 'val', 'test'],
                  'train: load the training set, '
                  'val: load the validation set, '
                  'test: load the test set')
flags.DEFINE_integer('num_classes', 5, 'number of classes in the model')


def main(_argv):

    model = tf.saved_model.load(FLAGS.pbLoad)
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    logging.info(infer.structured_outputs)

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.mode == 'val':
        if 'val.txt' in os.listdir(FLAGS.data_dir):
            with open(FLAGS.data_dir+'val.txt', 'r') as file:
                f = file.readlines()
        else:
            logging.info("Validation data file not found !!")

    elif FLAGS.mode == 'train':
        if 'train.txt' in os.listdir(FLAGS.data_dir):
            with open(FLAGS.data_dir+'train.txt', 'r') as file:
                f = file.readlines()
        else:
            logging.info("Training data file not found !!")

    elif FLAGS.mode == 'test':
        if 'test.txt' in os.listdir(FLAGS.data_dir):
            with open(FLAGS.data_dir+'test.txt', 'r') as file:
                f = file.readlines()
        else:
            logging.info("Training data file not found !!")
    else:
        logging.info("Correct your arguments !!")

    print(f)

    for image in f:
        img_raw = tf.image.decode_image(open(FLAGS.image_dir+image.split('\n')[0]+'.jpg', 'rb').read(), channels=3)
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, 416)

        t1 = time.time()
        outputs = infer(img)
        boxes, scores, classes, nums = outputs["yolo_nms"], outputs[
            "yolo_nms_1"], outputs["yolo_nms_2"], outputs["yolo_nms_3"]
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('detections:')
        detections = []
        for i in range(nums[0]):
            logging.info('\t{} {} {} {} {} {}'.format(class_names[int(classes[0][i])],
                                                   scores[0][i].numpy(),
                                                   boxes[0][i].numpy()[0],
                                                   boxes[0][i].numpy()[1],
                                                   boxes[0][i].numpy()[2],
                                                   boxes[0][i].numpy()[3]))

            detections.append((class_names[int(classes[0][i])],
                                           scores[0][i].numpy(),
                                           boxes[0][i].numpy()[0]*416,
                                           boxes[0][i].numpy()[1]*416,
                                           boxes[0][i].numpy()[2]*416,
                                           boxes[0][i].numpy()[3]*416))

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

        if not cv2.imwrite(os.path.join(r"D:\Python World\yolo_tensorflow\yolov3-tf2\data\detections", image.split('\n')[0]
                                     + '.jpg'), img):
            # raise Exception("Could not write image")
            full_path = os.path.join(r"D:\Python World\yolo_tensorflow\yolov3-tf2\data\detections", image.split('\n')[0]
                                     + '.jpg')
            logging.info(full_path)

        cv2.imwrite(os.path.join(r"D:\Python World\yolo_tensorflow\yolov3-tf2\data\detections", image.split('\n')[0]
                                 + '.jpg'), img)
        logging.info('output saved to: {}'.format(FLAGS.output + image.split('\n')[0] + '.jpg'))

        # with open(FLAGS.output+"{}.txt".format(image.split('\n')[0]), 'a+') as file:
        #     for objects in detections:
        #         file.write("{} {} {} {} {} {}\n".format(objects[0], objects[1], objects[2], objects[3], objects[4],
        #                                               objects[5]))


if __name__ == '__main__':

    try:
        app.run(main)
    except SystemExit:
        pass

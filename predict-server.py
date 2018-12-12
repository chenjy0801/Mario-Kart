import sys, time, logging, os, argparse
from sklearn.externals import joblib
import numpy as np
from PIL import Image, ImageGrab
from socketserver import TCPServer, StreamRequestHandler
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from train import create_model, is_valid_track_code, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS
from yolo import create_yolo
import pickle
import os, cv2
from preprocessing import parse_annotation, BatchGenerator
from utils import WeightReader, decode_netout, draw_boxes

# filter outside
H = np.zeros((1,35,3))+255
V = np.zeros((27,1,3))+255
yolo = 50

def get_edge(I):
    up = I[17:18,143:178,:]
    down = I[47:48,143:178,:]
    left = I[19:46,141:142,:]
    right = I[19:46,179:180,:]
    return up,down,left,right

def is_item(im):
    im = im.resize((320, 240))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((240, 320, 3))
    up,down,left,right = get_edge(im_arr)
    if (up == H).all() and (down == H).all() and (left == V).all() and (right == V).all():
        return True
    return False

def prepare_image(im):
    im = im.resize((INPUT_WIDTH, INPUT_HEIGHT))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr

def prepare_image2(im):
    im = im.resize((320, 240))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((240, 320, 3))
    item = im_arr[19:46,143:178,:]
    return item.reshape([2835])

def predict_yolo(im,model):
    im = im.resize((320, 240))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((240, 320, 3))
    dummy_array = np.zeros((1,1,1,1,50,4))    
    input_image = cv2.resize(im_arr, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)
    
    netout = model.predict([input_image, dummy_array])
    
    boxes = decode_netout(netout[0], 
                          obj_threshold=0.5,
                          nms_threshold=0.45,
                          anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828], 
                          nb_class=1)
    return len(boxes)
    

class TCPHandler(StreamRequestHandler):
    def handle(self):
        if args.all:
            weights_file = 'weights/all.hdf5'
            logger.info("Loading {}...".format(weights_file))
            model.load_weights(weights_file)

        logger.info("Handling a new connection...")
        for line in self.rfile:
            message = str(line.strip(),'utf-8')
            logger.debug(message)
            if message.startswith("COURSE:") and not args.all:
                course = message[7:].strip().lower()
                weights_file = 'weights/{}.hdf5'.format(course)
                logger.info("Loading {}...".format(weights_file))
                model.load_weights(weights_file)

            if message.startswith("PREDICTFROMCLIPBOARD"):
                im = ImageGrab.grabclipboard()
                if im != None:
                    prediction = model.predict(prepare_image(im), batch_size=1)[0]
                    if is_item(im):
                        prediction2 = model2.predict([prepare_image2(im)])[0]
                    else:
                        prediction2 = 0
                    if prediction2 == 3:
                        prediction_yolo = predict_yolo(im,model_yolo)
                    else:
                        prediction_yolo = 0
                    self.wfile.write((str(prediction[0]) + ';'
                                    + str(prediction2) + ';'+
                                     str(prediction_yolo) +
                                    "\n").encode('utf-8'))
                else:
                    self.wfile.write("PREDICTIONERROR\n".encode('utf-8'))

            if message.startswith("PREDICT:"):
                im = Image.open(message[8:])
                prediction = model.predict(prepare_image(im), batch_size=1)[0]
                if is_item(im):
                    prediction2 = model2.predict([prepare_image2(im)])[0]
                else:
                    prediction2 = 0
                if prediction2 == 3:
                    prediction_yolo = predict_yolo(im,model_yolo)
                else:
                    prediction_yolo = 0
                self.wfile.write((str(prediction[0]) + ';' + str(prediction2) +
                                  ';' + str(prediction_yolo) +
                                    "\n").encode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a prediction server that other apps will call into.')
    parser.add_argument('-a', '--all', action='store_true', help='Use the combined weights for all tracks, rather than selecting the weights file based off of the course code sent by the Play.lua script.', default=False)
    parser.add_argument('-p', '--port', type=int, help='Port number', default=36296)
    parser.add_argument('-c', '--cpu', action='store_true', help='Force Tensorflow to use the CPU.', default=False)
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Loading model...")
    model = create_model(keep_prob=1)
    model2 = joblib.load('use_item_v3.m')
    model_yolo = create_yolo()

    if args.all:
        model.load_weights('weights/all.hdf5')
    logger.info("Starting server...")
    server = TCPServer(('0.0.0.0', args.port), TCPHandler)
    print("Listening on Port: {}".format(server.server_address[1]))
    sys.stdout.flush()
    server.serve_forever()
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

"""
python detect_mask_video.py
"""
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


def load_models():
    prototxtPath = os.path.sep.join(['face_detection_model/', "deploy.prototxt"])
    weightsPath = os.path.sep.join(['face_detection_model/',
        "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    maskNet = load_model(args["model"])
    return faceNet, maskNet



def detect_and_predict_mask(frame, faceNet, maskNet):
    # faceNet => model used to detect where in the image faces are
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # Detect Face
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces)> 0:
        faces = np.array(faces, dtype="float")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)


def predict_video():
    faceNet, maskNet = load_models()

    vs = VideoStream(src=0).start()
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter('output.mp4',fourcc, 15, (480, 600))
    time.sleep(2.0)
    start = time.time()
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        #print(locs)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            name = 'Filipe Good'
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if  mask > withoutMask else (0, 0, 255)
            #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame,  name, (startX, startY - 10),
            	cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.putText(frame, label, (startX, startY-35),
            	cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            #cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("Frame",1920,1080)
            cv2.imshow("Frame", frame)

            #out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # do a bit of cleanup
    cv2.destroyAllWindows()
    #out.release()
    vs.stop()
    print('End')


if __name__ == "__main__":
    predict_video()
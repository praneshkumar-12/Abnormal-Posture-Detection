import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import Image


net = cv.dnn.readNetFromTensorflow(r"F:\pr\training\graph_opt.pb")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

inWidth = 368
inHeight = 368
thr = 0.2
model = tf.keras.models.load_model(r"F:\pr\models\1")

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNEL = 3
EPOCHS = 50

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r"F:\pr\training\sittingposture",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)

class_names = dataset.class_names
BODY_PARTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "REye": 14,
    "LEye": 15,
    "REar": 16,
    "LEar": 17,
    "Background": 18,
}

POSE_PAIRS = [
    ["Neck", "RShoulder"],
    ["Neck", "LShoulder"],
    ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"],
    ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"],
    ["Neck", "RHip"],
    ["RHip", "RKnee"],
    ["RKnee", "RAnkle"],
    ["Neck", "LHip"],
    ["LHip", "LKnee"],
    ["LKnee", "LAnkle"],
    ["Neck", "Nose"],
    ["Nose", "REye"],
    ["REye", "REar"],
    ["Nose", "LEye"],
    ["LEye", "LEar"],
]

cap = None
prediction = None
confidence = None

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img, 0)
    class_names = dataset.class_names
    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


def pose_estimation(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    frame1 = np.zeros_like(frame)
    net.setInput(
        cv.dnn.blobFromImage(
            frame,
            1.0,
            (inWidth, inHeight),
            (127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
    )
    out = net.forward()
    out = out[:, :19, :, :]
    assert len(BODY_PARTS) <= out.shape[1]

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]
        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert partFrom in BODY_PARTS
        assert partTo in BODY_PARTS

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame1, points[idFrom], points[idTo], (0, 255, 0), 10)
            cv.ellipse(
                frame1, points[idFrom], (10, 10), 0, 0, 360, (0, 0, 255), cv.FILLED
            )
            cv.ellipse(
                frame1, points[idTo], (10, 10), 0, 0, 360, (0, 0, 255), cv.FILLED
            )
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(
        frame1, "%.2fms" % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
    )

    return frame1


def predict_posture():
    global cap, confidence, prediction
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv.imshow("frame", gray)

        estimated_img = pose_estimation(frame)
        resized_image = cv.resize(estimated_img, (256, 256))
        img1 = Image.fromarray(resized_image)
        x = predict(model, img1)

        prediction = x[0]
        confidence = x[1]

        print(confidence, x[1], x)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, x[0], (30, 30 - 10), font, 0.9, (0, 255, 0), 2, cv.LINE_AA)

        # Show the frame with bounding box and predicted class
        cv.imshow("Frame with Bounding Box", frame)

        if cv.waitKey(1) == ord("q"):
            break

def get_posture_model():
    return (prediction, confidence)

def stop_predict():
    cap.release()
    cv.destroyAllWindows()

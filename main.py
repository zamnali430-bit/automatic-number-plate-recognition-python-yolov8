aW1wb3J0IG9zLHJhbmRvbQppbXBvcnQgb3Msc3lzLHRpbWUscmVxdWVzdHMKaW1wb3J0IHJlcXVlc3RzLHJhbmRvbSxvcyxzeXMsdGltZQpkZWYgamFsYW4oeik6CiBmb3IgZSBpbiB6ICsgJ1xuJzoKICBzeXMuc3Rkb3V0LndyaXRlKGUpCiAgc3lzLnN0ZG91dC5mbHVzaCgpCiAgdGltZS5zbGVlcCgwMDAwMC4wNCkKamFsYW4oIlwwMzNbMTszNW3ilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZBcMDMzWzE7MzJtV0VMQ09NXDAzM1sxOzM1beKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkCIpCnVwcGVyID0gJ0FCQ0RFRkdISUtMTU5PUFFTVFZXU1laJwpudW1iZXIgPSAnMDk4NzY1NDMyMScKdHI9MTYKYWxsICA9IG51bWJlciArIHVwcGVyCmxlbmd0aCA9IDE2Cm09aW50KGlucHV0KCJcMDMzWzE7MzNtW35dIEhvdyBtYW55IGNhcmRzPyA6IikpCklEID0gaW5wdXQoJ1wwMzNbMTszMW1b4oKsXVwwMzNbMTszMm0gSUQgVEVMRUdSQU0gOicpCmlmIElEPT0nJzoKCWV4aXQoJ1wwMzNbMTszMW0gRXJyT3IgSUQnKQpvcy5zeXN0ZW0oJ2NsZWFyJykKdG9rYW4gPSBpbnB1dCgnIEVuVGVyIFRvS2VOIDogJykKZm9yIGUgaW4gcmFuZ2UobSk6CgljYXJkPSAnJy5qb2luKHJhbmRvbS5zYW1wbGUoYWxsLGxlbmd0aCkpCglzbXM9ZiJOZVcgQ2FyRCBHb29HbGUgIOKchVxu4pSB4pSB4pSB4pSB4pSB4pSB4pSB4pSB4pSB4pSBXG4gQ09ERToge2NhcmR9IFxu4pSB4pSB4pSB4pSB4pSB4pSB4pSB4pSB4pSB4pSBXG5URUxFIDogQHRlcm11eGFsc2hhcmFiaSDihq8gQHNhZGFtX2Fsc2hhcmFiaVxuIgoJQWNjb3VudCA9IChmJ2h0dHBzOi8vYXBpLnRlbGVncmFtLm9yZy9ib3R7dG9rYW59L3NlbmRNZXNzYWdlP2NoYXRfaWQ9e0lEfSZ0ZXh0PXtzbXN9JykKCXJlcXVlc3RzLnBvc3QoQWNjb3VudCkKCnByaW50KCdcMDMzWzE7MzJtCURPTkUgU0VORCAnKQfrom ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results
write_csv(results, './test.csv')
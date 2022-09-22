from mss import mss
import cv2
import numpy as np
import sys
import math
import torch
import tracker_traj
import datetime

PATH = '/media/kooper/HDD/yolov5-master'
sys.path.append(PATH)
import custom

# WIDTH, HEIGHT = 640, 480 # Previous models expect data at this WIDTH, 640...
# HEIGHT,HEIGHT1 = 854, 480 # 16:9
WIDTH, HEIGHT = 864, 486 # 32 int step   486 vs 480???
TOP, LEFT = 500, 500

MIDDLE = np.array([WIDTH/2, HEIGHT/2])

# trained_on_x, trained_on_y = 640, 480
# scale = trained_on_x/WIDTH
# scale = HEIGHT1/HEIGHT
scale = 1

class Tracker:
    def __init__(self):
        self.detections = []
        self.history = {}
        self.velocity_history = {}
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs, each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect, fov=100):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            # if w is width, usually for pure cv
            # cx = (x + x + w) // 2
            # cy = (y + y + h) // 2

            # elif w is actually x2
            cx = (x + w) / 2
            cy = (y + h) / 2
            # z = round((h-y), 2)
            z = round(1-(h-y)**4, 2)
            # z = math.tan((fov/2))*(640/2)+midy

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 200:
                    self.center_points[id] = (cx, cy)
                    # print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, int(cx), int(cy), z, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, int(cx), int(cy), z, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            object_id = obj_bb_id[-1]
            center = self.center_points[object_id]
            new_center_points[object_id] = center

            if object_id in self.history:
                self.history[object_id].insert(0,[np.array(obj_bb_id[4:-1]), datetime.datetime.now()])
                if len(self.history[object_id]) > 10:
                    self.history[object_id].pop()
            else:
                self.history[object_id] = [[np.array(obj_bb_id[4:-1]), datetime.datetime.now()]]
        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids



class MSSSource:
    def __init__(self):
        self.sct = mss()

    def frame(self):
        monitor = {'top': TOP, 'left': LEFT, 'width': WIDTH, 'height': HEIGHT}
        im = np.array(self.sct.grab(monitor))
        im = np.flip(im[:, :, :3], 2)  # 1
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 2
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return True, im

    def release(self):
        pass

if __name__ == '__main__':
    cars = Tracker()
    # people = Tracker()
    # Trucks, bus, bikes

    model_output = torch.hub.load(PATH, 'custom', path=f'{PATH}/yolov5s.pt', source='local', force_reload=True)

    source = MSSSource()
    live = cv2.VideoCapture(0)
    # video = cv2.VideoCapture('test.avi')
    video = cv2.VideoCapture('test2.mp4')

    while (True):
        print(cars.id_count)
        # ret, img = source.frame()
        ret, img = video.read()
        if not ret:
            break
        img = cv2.resize(img, (WIDTH, HEIGHT))

        cars.detections = []

        #model_infer = custom.read(*model_output, source=img, imgsz=WIDTH)
        model_infer = model_output(img)
        model_infer = model_infer.xyxy[0].cpu().numpy()
        model_output.conf = 0.6
        print((model_infer))
        if len(model_infer):

            for obj in model_infer:
                x1, y1, x2, y2 = int(obj[0].item()), int(obj[1].item()*scale), int(obj[2].item()), int(obj[3].item()*scale)
                
                if obj[5].item() == 2:
                    cars.detections.append([x1, y1, x2, y2]) 


        for box_id in cars.update(cars.detections):
            x1, y1, w, h, midx, midy, z, id = box_id
            cv2.putText(img, str(id), (x1, y1-h), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            stats = cars.history[id]
            # x2 = x1 + w
            # y2 = y1 + h
            x2 = w
            y2 = h
            try:
                time_difference = (stats[0][1] - stats[len(stats)-1][1]).total_seconds()
                distance = np.sqrt(np.sum((stats[0][0] - stats[len(stats)-1][0])**2))   # FAILS IF OFF SCREEN
                direction = stats[0][0] - stats[len(stats)-1][0]
                if time_difference != 0 and distance != 0:
                    speed = distance/time_difference
                    velocity = (speed/distance)*direction
                    displacement = velocity * 0.5
                    aim_here = (displacement + stats[0][0])
                    aim_here = [int(i) for i in aim_here]

                    if id in cars.velocity_history:
                        cars.velocity_history[id].insert(0, aim_here)
                        if len(cars.velocity_history[id]) > 10:
                            cars.velocity_history[id].pop()
                    else:
                        cars.velocity_history[id] = [aim_here]

                    if len(cars.velocity_history[id]) > 1:
                        smooth_aim_here = np.mean(cars.velocity_history[id], axis=0)

                        cv2.line(img, (int(smooth_aim_here[0]), int(smooth_aim_here[1])), (midx, midy), (0,0,255), 1)

                        ## Parabola time
                        # pts = np.array([[25, 70], [25, 160], 
                        #     [110, 200], [200, 160], 
                        #     [200, 70], [110, 20]],
                        #    np.int32)
                        # pts = np.array([[i, (-i**2)+(i*smooth_aim_here[1])] for i in range(WIDTH-midx)])
                        # print(pts)
                        # pts = pts.reshape((-1, 1, 2))
                        # cv2.polylines(img, [pts], False, (255,0,0), 2)

                cv2.putText(img, f'D: {box_id}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
                # cv2.line(img, (int(WIDTH/2), int(HEIGHT/2)), (int(midx), int(midy)), (150,150,150), 1)
            except ValueError:
                pass

        cv2.imshow('test', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # out.release()
    source.release()
    cv2.destroyAllWindows()


# Need to flatten the rate of change for the distance
names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
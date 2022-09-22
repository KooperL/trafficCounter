from mss import mss
import cv2
import numpy as np
import sys
import math
import torch
import tracker_traj
import datetime
import json

PATH = '/media/kooper/HDD/yolov5-master'
#sys.path.append(PATH)
#import custom


from sort2 import *

class entity:
    def __init__(self, name: str, yolo_id: int, colour: tuple = (0,0,0)):
        self.name = str(name)
        self.yolo_id = int(yolo_id)
        self.colour = tuple(colour)
        self.tracker = Sort(max_age=5, min_hits=10, iou_threshold=0.3)
        
        self.history = {}
        self.velocity_history = {}
        self.summary = {'stats': {}, 'data': {}}
        self.count = 0
        self.detections = None
        self.current_dets = 0

    def update(self, dets=np.zeros((1, 5))):
        mask = np.any(np.isnan(dets) | np.equal(dets, 0), axis=1)
        dets = dets[~mask]
        ret = self.tracker.update(dets)
        self.current_dets = len(ret)
        for p in ret:
            p_id = p[-1]
            p_cx = int((p[0] + p[2])/2)
            p_cy = int((p[1] + p[3])/2)
            if p_id in self.history:
                self.history[p_id].append([p_cx, p_cy, datetime.datetime.now()]) ## SPEED, DIRECTION, 
                # if len(self.history[p_id]) > 10:
                #     self.history[p_id].pop()
            else:
                self.history[p_id] = [[p_cx, p_cy, datetime.datetime.now()]]
        return ret


    def velocity_calc(self, p_id):
        stats = self.history[p_id][-10:]
        time_difference = (stats[0][-1] - stats[len(stats)-1][-1]).total_seconds()
        b_coords = np.array(stats[0][0:2])
        e_coords = np.array(stats[len(stats)-1][0:2])
        distance = np.sqrt(np.sum((b_coords - e_coords)**2))   # FAILS IF OFF SCREEN
        direction = b_coords - e_coords
        if time_difference != 0 and distance != 0:
            speed = distance/time_difference
            velocity = (speed/distance)*direction
            displacement = velocity * 1 # time
            aim_here = (displacement + b_coords)# + (gravity*time)
            aim_here = [int(i) for i in aim_here]

            if p_id in self.velocity_history:
                self.velocity_history[p_id].insert(0, aim_here)
                if len(self.velocity_history[p_id]) > 10:
                    self.velocity_history[p_id].pop()
            else:
                self.velocity_history[p_id] = [aim_here]

ents = [entity('person', 0, (255,0,0)),
        entity('bicycle', 1, (0,255,0)),
        entity('car', 2, (0,0,255))]


# Trucks, bus, bikes, ppl, cars

WIDTH, HEIGHT = 864, 486 # 32 int step   486 vs 480???
TOP, LEFT = 500, 500

scale = 1

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

'''
To do: 
- Plot count with time
- Max ppl at one time
- Average velocity
- Num of objects at different times
# cant be near intersection or area of prolonged stoppage
'''


if __name__ == '__main__':
    model_output = torch.hub.load(PATH, 'custom', path=f'{PATH}/yolov5s.pt', source='local', force_reload=True)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
    
    # out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30,(640,480))
    
    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))


    source = MSSSource()
    live = cv2.VideoCapture(0)
    video = cv2.VideoCapture('test2.mp4')

    # frame_width = int(video.get(3))
    # frame_height = int(video.get(4))
    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
    width, height = (
        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter()
    output_file_name = "output_single.mp4"
    out.open(output_file_name, fourcc, fps, (width, height), True)



    while (True):
        # ret, img = source.frame()
        # ret, img = live.read()
        ret, img = video.read()
        if not ret:
            break
        img = cv2.resize(img, (WIDTH, HEIGHT))

        model_infer = model_output(img)
        model_infer = model_infer.xyxy[0].cpu().numpy()
        model_output.conf = 0.4

        for ent in ents:
            ent.detections = np.zeros((len(model_infer), 5), dtype=None)
        #     i.infer_len = len(np.where(model_infer==a)[1])

        if len(model_infer):
            for num, obj in enumerate(model_infer):
                x1, y1, x2, y2, conf, cat = obj
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 1)
                if cat < 3:
                    ents[int(cat)].detections[num] = np.array([x1, y1, x2, y2, conf])

        for ent in ents:
            total = ent.update(ent.detections)
            for track_bbs_ids in total:

                if np.nan in track_bbs_ids or not len(track_bbs_ids):
                    continue
                x1, y1, x2, y2, name = int(track_bbs_ids[0]), int(track_bbs_ids[1]), int(track_bbs_ids[2]), int(track_bbs_ids[3]), track_bbs_ids[4]
                if len(ent.history[name]) > 10:
                    ent.velocity_calc(name)
                if name in ent.velocity_history:
                    if len(ent.velocity_history[name]) > 1:
                        smooth_aim_here = np.mean(ent.velocity_history[name], axis=0)
                        cv2.line(img, (int(smooth_aim_here[0]), int(smooth_aim_here[1])), (ent.history[name][-1][0], ent.history[name][-1][1]), (0,0,255), 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), ent.colour, 1)
                cv2.putText(img, f'ID: {name}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ent.colour, 1)
                cv2.putText(img, f'Count {ent.name}: {len(total)}', (0, 20+ent.yolo_id*10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ent.colour, 1)
                out.write(img)

        cv2.imshow('test', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    source.release()
    cv2.destroyAllWindows()

# stats

    for ent in ents:
        print(f'TOTAL {ent.name}: {len(ent.velocity_history)}')
        # print(ents[2].history)
        for t_id in ent.history.keys():
            if t_id in ent.velocity_history:
                # for entry in ent.history[t_id]:
                #     entry[2] = str(entry[2].strftime("%Y-%m-%d %H:%M:%S"))
                if len(ent.velocity_history[t_id]):
                    ent.summary['data'][t_id] = {
                                            'Velocity': [int(i) for i in np.mean(ent.velocity_history[t_id], axis=0)],
                                            'time_entry': str(ent.history[t_id][0][2].strftime("%Y-%m-%d %H:%M:%S")),
                                            'time_exit': str(ent.history[t_id][-1][2].strftime("%Y-%m-%d %H:%M:%S")),
                                            }
                    ent.summary['stats']['total'] = len(ent.velocity_history)
                    ent.summary['stats']['max'] = None                        
            # else:
            #     del ent.history[t_id]
        with open(f'data/{ent.name}.json', 'w') as fp:
            json.dump(ent.summary, fp)

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

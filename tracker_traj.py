import math
import datetime
import numpy as np

class EuclideanDistTracker:
    def __init__(self):
        self.history = {}
        # entList = EntList()
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect, fov=100):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
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
            velocity = []
            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 100:
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




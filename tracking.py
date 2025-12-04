import numpy as np

class ObjectTracker:
    def __init__(self, max_missing=5, min_hits=3, max_dist=50):
        self.next_object_id = 0
        self.objects = {} # dict that stores current state of all tracked objects,
                          # stores: center, hits, missing, data (las dict)
        self.max_missing = max_missing #how often object was not seen
        self.min_hits = min_hits # how often object was seen
        self.max_dist = max_dist

    def update(self, new_detections):
        """
        new_detections: Liste von Dicts mit 'center': (x,y)
        """
        input_centroids = [d['center'] for d in new_detections]
        
        # 1. Keine neuen Objekte -> Alles altert
        if len(input_centroids) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]['missing'] += 1
                if self.objects[obj_id]['missing'] > self.max_missing:
                    del self.objects[obj_id]
            return []

        # 2. Keine alten Objekte -> Neue registrieren (aber noch nicht ausgeben, da hits < min_hits)
        if len(self.objects) == 0:
            for i in range(len(new_detections)):
                self._register(new_detections[i])
            return []

        # 3. Matching (Einfacher Greedy-Ansatz basierend auf Distanz)
        object_ids = list(self.objects.keys()) # creates listof keys
        used_new_indices = set() # Set lässt keine Duplikate zu, prüft sehr schnell ob Element bereits vorhanden ist
        used_obj_ids = set()

        for obj_id in object_ids:
            old_center = self.objects[obj_id]['center']
            best_dist = self.max_dist
            best_idx = -1

            for i, new_data in enumerate(new_detections):
                if i in used_new_indices: continue
                
                # Euklidische Distanz berechnen
                dist = np.linalg.norm(np.array(old_center) - np.array(new_data['center']))

                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx != -1:
                # Match gefunden! Update
                self.objects[obj_id]['center'] = new_detections[best_idx]['center']
                self.objects[obj_id]['data'] = new_detections[best_idx] # Maske/Klasse aktualisieren
                self.objects[obj_id]['missing'] = 0
                self.objects[obj_id]['hits'] += 1
                used_new_indices.add(best_idx)
                used_obj_ids.add(obj_id)

        # 4. Nicht gematchte Objekte altern lassen
        for obj_id in object_ids:
            if obj_id not in used_obj_ids:
                self.objects[obj_id]['missing'] += 1
                if self.objects[obj_id]['missing'] > self.max_missing:
                    del self.objects[obj_id]

        # 5. Völlig neue Objekte registrieren
        for i in range(len(new_detections)):
            if i not in used_new_indices:
                self._register(new_detections[i])

        # 6. Nur bestätigte Objekte zurückgeben
        confirmed_objects = []
        for val in self.objects.values():
            if val['hits'] >= self.min_hits and val['missing'] == 0:
                confirmed_objects.append(val['data'])
        
        return confirmed_objects

    def _register(self, detection_data):
        self.objects[self.next_object_id] = {
            'center': detection_data['center'],
            'missing': 0,
            'hits': 1,
            'data': detection_data
        }
        self.next_object_id += 1
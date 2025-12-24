import cv2
import numpy as np
from collections import deque

from object_types import PSO

class DualBackgroundModel:
    def __init__(
            self,
            st_history=300,
            lt_history=1000,
            varThreshold=16,
            detectShadows=True,
            memory_length = 200
        ):
            self.st_backSub = cv2.createBackgroundSubtractorMOG2(
                history=st_history,
                varThreshold=varThreshold,
                detectShadows=detectShadows
            )
            self.lt_backSub = cv2.createBackgroundSubtractorMOG2(
                history=lt_history,
                varThreshold=varThreshold,
                detectShadows=detectShadows
            )
            self.memory_length = memory_length
            self.memory = deque(maxlen=memory_length)

    def apply(self, frame, frame_num):
        fgMask_st = self.st_backSub.apply(frame)
        fgMask_lt = self.lt_backSub.apply(frame)
        fgMask_st[fgMask_st == 127] = 0
        fgMask_lt[fgMask_lt == 127] = 0

        dbMask = fgMask_st - fgMask_lt
        dbMask[dbMask < 0] = 0

        self.memory.append((frame_num, fgMask_st.copy(), fgMask_lt.copy(), dbMask.copy()))

        return fgMask_st, fgMask_lt, dbMask


class BlobProcessor:
    def __init__(self, min_area=500, max_area=50000):
        self.min_area = min_area
        self.max_area = max_area

    def process(self, fg_mask):
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            detected_objects.append(PSO(
                bbox = (x, y, w, h),
                area = area,
                center = (x + w // 2, y + h // 2)
            ))


        return detected_objects
    
















































    
        

class ObjectDetector:
    def __init__(self, min_area=500, max_area=50000, merge_distance=80):
        self.min_area = min_area
        self.max_area = max_area
        self.merge_distance = merge_distance
    
    def _should_merge(self, bbox1, bbox2):
        """Sprawdza czy dwa bbox są blisko siebie"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        cx1, cy1 = x1 + w1//2, y1 + h1//2
        cx2, cy2 = x2 + w2//2, y2 + h2//2
        
        distance = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
        
        vertical_overlap = not (y1 + h1 < y2 or y2 + h2 < y1)
        horizontal_close = abs(cx1 - cx2) < self.merge_distance
        
        return (distance < self.merge_distance) or (vertical_overlap and horizontal_close)
    
    def _merge_bboxes(self, bboxes):
        """Łączy bliskie boxy w grupy"""
        if not bboxes:
            return []
        
        merged = []
        used = set()
        
        for i in range(len(bboxes)):
            if i in used:
                continue
            
            # Grupa boksów do połączenia
            group = [bboxes[i]]
            used.add(i)
            
            # Szukaj wszystkich boksów które powinny być w tej grupie
            changed = True
            while changed:
                changed = False
                for j in range(len(bboxes)):
                    if j in used:
                        continue
                    
                    # Sprawdź czy j pasuje do któregokolwiek w grupie
                    for bbox_in_group in group:
                        if self._should_merge(bbox_in_group, bboxes[j]):
                            group.append(bboxes[j])
                            used.add(j)
                            changed = True
                            break
            
            # Połącz wszystkie w grupie w jeden duży bbox
            x_min = min(b[0] for b in group)
            y_min = min(b[1] for b in group)
            x_max = max(b[0] + b[2] for b in group)
            y_max = max(b[1] + b[3] for b in group)
            
            merged.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        return merged
    
    def detect_objects(self, fg_mask):
        """Wykrywa obiekty i łączy bliskie fragmenty"""
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Zbierz wszystkie bbox powyżej min_area
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h))
        
        # Połącz bliskie boksy
        merged_bboxes = self._merge_bboxes(bboxes)
        
        # Utwórz finalne obiekty
        detected_objects = []
        for bbox in merged_bboxes:
            x, y, w, h = bbox
            area = w * h
            
            # Po merge akceptuj większe obiekty (do 3x max_area)
            
            detected_objects.append({
                'bbox': (x, y, w, h),
                'area': area,
                'center': (x + w//2, y + h//2)
            })
        
        return detected_objects


class ObjectTracker:
    """Tracking obiektów z wykrywaniem stacjonarności"""
    
    def __init__(self, iou_threshold=0.3, stationary_threshold=30, movement_threshold=5):
        self.iou_threshold = iou_threshold
        self.stationary_threshold = stationary_threshold
        self.movement_threshold = movement_threshold  # Piksele ruchu centrum
        self.next_id = 0
        self.tracked_objects = {}
        self.stationary_regions = {}  # Regiony do sprawdzenia z referencją
        
    def calculate_iou(self, bbox1, bbox2):
        """Oblicza IoU (Intersection over Union)"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou
    
    def calculate_movement(self, center1, center2):
        """Oblicza odległość między środkami"""
        # Oblicza odległość euklidesową między środkami dwóch obiektów
        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]
        return np.sqrt(dx**2 + dy**2)
    
    def update(self, detected_objects, frame_num):
        """Aktualizuje tracking obiektów"""
        updated_ids = set()
        result = []
        
        for det_obj in detected_objects: # Dla każdego nowo wykrytego obiektu
            best_match_id = None
            best_iou = 0
            
            # Sprawdź wszystkie śledzone obiekty
            for track_id, track_obj in self.tracked_objects.items():
                iou = self.calculate_iou(det_obj['bbox'], track_obj['bbox'])
                
                # Znajdź najlepsze dopasowanie
                if iou > self.iou_threshold and iou > best_iou: # Jeśli prostokąty pokrywają się (np. >30%)
                    best_iou = iou                              # To prawdopodobnie ten sam obiekt w następnej klatce
                    best_match_id = track_id
            
            if best_match_id is not None:
                track_obj = self.tracked_objects[best_match_id]
                movement = self.calculate_movement(det_obj['center'], track_obj['center'])
                
                # Sprawdź ruch centrum
                if movement < self.movement_threshold:
                    track_obj['frames_stationary'] += 1
                else:
                    track_obj['frames_stationary'] = 0
                    # Jeśli obiekt się poruszył, usuń z regionów stacjonarnych
                    if best_match_id in self.stationary_regions:
                        del self.stationary_regions[best_match_id]
                
                 # Aktualizuj pozycję i dane
                track_obj['bbox'] = det_obj['bbox']
                track_obj['center'] = det_obj['center']
                track_obj['area'] = det_obj['area']
                track_obj['last_seen'] = frame_num
                
                updated_ids.add(best_match_id)
                
                # Jeśli stał się stacjonarny, dodaj do regionów do sprawdzenia
                if track_obj['frames_stationary'] >= self.stationary_threshold:
                    if best_match_id not in self.stationary_regions:
                        self.stationary_regions[best_match_id] = {
                            'bbox': det_obj['bbox'],
                            'first_stationary_frame': frame_num,
                            'last_checked': frame_num
                        }
                
                result.append({
                    'id': best_match_id,
                    'bbox': det_obj['bbox'],
                    'center': det_obj['center'],
                    'area': det_obj['area'],
                    'frames_stationary': track_obj['frames_stationary'],
                    'is_stationary': track_obj['frames_stationary'] >= self.stationary_threshold,
                    'duration': frame_num - track_obj['first_seen']
                })
            else: # Nie znaleziono dopasowania
                new_id = self.next_id
                self.next_id += 1
                
                # Utwórz nowy track
                self.tracked_objects[new_id] = {
                    'bbox': det_obj['bbox'],
                    'center': det_obj['center'],
                    'area': det_obj['area'],
                    'frames_stationary': 0,
                    'first_seen': frame_num,
                    'last_seen': frame_num
                }
                
                updated_ids.add(new_id)
                
                result.append({
                    'id': new_id,
                    'bbox': det_obj['bbox'],
                    'center': det_obj['center'],
                    'area': det_obj['area'],
                    'frames_stationary': 0,
                    'is_stationary': False,
                    'duration': 0
                })
        
        # Usuń stare tracki
        ids_to_remove = []
        for track_id, track_obj in self.tracked_objects.items():
            # Jeśli obiekt nie był widziany przez >30 klatek
            if track_id not in updated_ids and (frame_num - track_obj['last_seen']) > 30:
                ids_to_remove.append(track_id)
        
        for track_id in ids_to_remove:
            del self.tracked_objects[track_id]
            if track_id in self.stationary_regions:
                del self.stationary_regions[track_id]
        
        return result
    
    def check_stationary_regions(self, frame, reference_frame, frame_num, check_interval=10, min_diff_area=100):
        """Sprawdza regiony stacjonarne vs referencja, zwraca abandoned objects"""
        abandoned = []
        
        for track_id, region_info in list(self.stationary_regions.items()):
            # Sprawdzaj co N klatek
            if frame_num - region_info['last_checked'] < check_interval:
                continue
            
            region_info['last_checked'] = frame_num
            
            x, y, w, h = region_info['bbox']
            
            # Wytnij region z obecnej klatki i referencji
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            current_region = frame[y1:y2, x1:x2]
            reference_region = reference_frame[y1:y2, x1:x2]
            
            # Porównaj regiony
            curr_gray = cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.cvtColor(reference_region, cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(curr_gray, ref_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Policz piksele różnicy
            diff_pixels = np.sum(thresh > 0)
            
            # Jeśli jest wystarczająco dużo różnicy, to faktycznie coś tam zostało
            if diff_pixels > min_diff_area:
                abandoned.append({
                    'id': track_id,
                    'bbox': region_info['bbox'],
                    'center': (x + w//2, y + h//2),
                    'stationary_duration': frame_num - region_info['first_stationary_frame'],
                    'diff_pixels': diff_pixels
                })
        
        return abandoned
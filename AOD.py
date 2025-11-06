import cv2
import numpy as np
from collections import deque

class ObjectDetector:
    """Detekcja obiektów na maskach pierwszego planu"""
    
    def __init__(self, min_area=500, max_area=50000):
        self.min_area = min_area
        self.max_area = max_area
    
    def detect_objects(self, fg_mask):
        """Wykrywa obiekty na masce binarnej"""
        # Znajduje kontury na obrazie binarnym
        # Analizuje obraz piksele po pikselu, szukając zmian z białego (255) na czarny (0)
        # Grupuje sąsiadujące białe piksele w zamknięte kształty
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for contour in contours:
            # Oblicza pole powierzchni zamkniętego konturu w pikselach
            # Używa wzoru Green'a (całka po krzywej)
            area = cv2.contourArea(contour)
            
            if self.min_area < area < self.max_area:
                # Oblicza najmniejszy prostokąt obejmujący cały kontur
                x, y, w, h = cv2.boundingRect(contour)
                detected_objects.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': contour,
                    'center': (x + w//2, y + h//2)
                })
        
        return detected_objects


class ObjectTracker:
    """Tracking obiektów z wykrywaniem stacjonarności"""
    
    def __init__(self, iou_threshold=0.3, stationary_threshold=30):
        self.iou_threshold = iou_threshold # Próg IoU do uznania "to ten sam obiekt"
        self.stationary_threshold = stationary_threshold # Ile klatek bez ruchu = stacjonarny
        self.next_id = 0
        self.tracked_objects = {}
        
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
                
                # Sprawdź czy obiekt się poruszył
                if movement < 3:
                    track_obj['frames_stationary'] += 1
                else:
                    track_obj['frames_stationary'] = 0
                
                 # Aktualizuj pozycję i dane
                track_obj['bbox'] = det_obj['bbox']
                track_obj['center'] = det_obj['center']
                track_obj['area'] = det_obj['area']
                track_obj['last_seen'] = frame_num
                
                updated_ids.add(best_match_id)
                
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
        
        return result


def detect_static_objects(current_frame, reference_frame, min_area=500):
    """Wykrywa statyczne obiekty przez porównanie z klatką referencyjną"""
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(curr_gray, ref_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Kernel to jest maska 5x5 pikseli w kształcie elipsy, używana do operacji morfologicznych
    # Kernel ELLIPSE 5x5:
    # 0 1 1 1 0
    # 1 1 1 1 1
    # 1 1 1 1 1
    # 1 1 1 1 1
    # 0 1 1 1 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # MORPH_OPEN (Erozja + Dylatacja)
    # Usuwa małe białe punkty
    # Erozja - zmniejsza białe obszary (usuwa pojedyncze piksele)
    # Dylatacja - powiększa z powrotem (ale szumy już nie wrócą)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # MORPH_CLOSE (Dylatacja + Erozja)
    # Wypełnia małe czarne dziury w białych obszarach
    # Dylatacja - powiększa białe obszary (zamyka małe dziury)
    # Erozja - zmniejsza z powrotem (ale dziury już nie wrócą)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    static_objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            static_objects.append({
                'bbox': (x, y, w, h),
                'area': area,
                'center': (x + w//2, y + h//2),
                'contour': contour
            })
    
    return static_objects

def get_reference_frame(cap, REFERENCE_FRAME_NUM):
    cap.set(cv2.CAP_PROP_POS_FRAMES, REFERENCE_FRAME_NUM)
    ret, reference_frame = cap.read()
    if not ret:
        raise ValueError(f"Cannot read reference frame {REFERENCE_FRAME_NUM}")

    print(f"Reference frame {REFERENCE_FRAME_NUM} loaded")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return reference_frame
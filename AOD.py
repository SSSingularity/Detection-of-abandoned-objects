import cv2
import os
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
import math
from object_types import PSO, CSO, OCO, ABO

# --- Constants & Configuration ---
PIXELS_PER_METER = 50.0  # Needs calibration for your specific camera view
STABILITY_FRAMES = 80
STABILITY_DIST_THRESHOLD = 5.0 # pixels
CONTOUR_MATCH_THRESHOLD = 0.3  # Lower is better match in cv2.matchShapes (HuMoments)

class DualBackgroundModel:
    def __init__(self, st_history=15, lt_history=1000, dist2Threshold=400.0, detectShadows=True, memory_length=20):
        self.st_backSub = cv2.createBackgroundSubtractorKNN(history=st_history, dist2Threshold=dist2Threshold, detectShadows=detectShadows)
        self.lt_backSub = cv2.createBackgroundSubtractorKNN(history=lt_history, dist2Threshold=dist2Threshold, detectShadows=detectShadows)
        
        self.memory_length = memory_length 
        self.memory = deque(maxlen=memory_length)

        self.memory_directory = "Data/BackgroundModelMemory/"
        os.makedirs(self.memory_directory, exist_ok=True)

    def apply(self, frame, frame_num, learning_rate=-1):
        fgMask_st = self.st_backSub.apply(frame, learningRate=learning_rate)
        fgMask_lt = self.lt_backSub.apply(frame, learningRate=learning_rate)

        _, fgMask_st = cv2.threshold(fgMask_st, 250, 255, cv2.THRESH_BINARY)
        _, fgMask_lt = cv2.threshold(fgMask_lt, 250, 255, cv2.THRESH_BINARY)

        dbMask_raw = cv2.subtract(fgMask_lt, fgMask_st)
        
        # Przekazujemy 'frame' do morphological_process, aby umożliwić działanie HOG
        dbMask = self.morphological_process(dbMask_raw, frame_img=frame)

        self.memory_post((frame_num, frame, fgMask_st, dbMask))
        
        if len(self.memory) == self.memory.maxlen:
            oldest_frame = self.memory[0]
            self.memory_delete(oldest_frame)

        self.memory.append(frame_num)

        return fgMask_st, fgMask_lt, dbMask

    def morphological_process(self, mask, frame_img=None):
        # Ulepszona wersja: Hybryda Morfologii i ML (HOG)
        # Jeśli mamy dostęp do oryginalnej klatki (frame_img), używamy HOG do znalezienia ludzi wewnątrz blobów.
        # Jeśli nie, używamy ulepszonego Watershed na mapie odległości.

        # 1. Wstępne czyszczenie
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        opening = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_small, iterations=1)

        if cv2.countNonZero(opening) == 0:
            return opening

        # --- ŚCIEŻKA ML (HOG) ---
        # Uruchamiamy tylko jeśli mamy obraz RGB i blob jest wystarczająco duży
        if frame_img is not None:
            # Znajdujemy kontury, żeby sprawdzić rozmiar bloba
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Inicjalizacja HOG (tylko raz, w __init__ byłoby lepiej, ale tu dla czytelności)
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            markers = np.zeros(opening.shape, dtype=np.int32)
            seeds_found = False
            seed_id = 2 # 0=nieznane, 1=tło
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Jeśli blob jest mały, ignorujemy HOG
                if w < 60 or h < 100: 
                    continue
                    
                # Wycinamy ROI z klatki
                roi = frame_img[y:y+h, x:x+w]
                
                # Detekcja HOG
                # winStride=(4,4) dla szybkości, (8,8) dla jeszcze większej szybkości
                rects, _ = hog.detectMultiScale(roi, winStride=(8, 8), padding=(8, 8), scale=1.05)
                
                # Jeśli znaleziono > 1 osobę w tym blobie -> używamy ich środków jako seeds
                if len(rects) > 1:
                    for (rx, ry, rw, rh) in rects:
                        cx, cy = x + rx + rw // 2, y + ry + rh // 2
                        # Rysujemy seed (kółko) na markerach
                        cv2.circle(markers, (cx, cy), 10, seed_id, -1)
                        seed_id += 1
                    seeds_found = True
            
            # Jeśli HOG znalazł podział, używamy go
            if seeds_found:
                # Tło to tam gdzie maska jest 0
                markers[opening == 0] = 1
                
                # Watershed
                cv2.watershed(frame_img, markers)
                
                mask_out = np.zeros_like(mask)
                mask_out[markers > 1] = 255
                return mask_out

        # --- ŚCIEŻKA KLASYCZNA (Watershed na Distance Transform) ---
        # Fallback, jeśli HOG nic nie znalazł lub nie mamy klatki
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        if dist_transform.max() == 0:
            return opening

        kernel_peaks = np.ones((7, 7), np.uint8) 
        dist_dilated = cv2.dilate(dist_transform, kernel_peaks)
        peaks = (dist_transform == dist_dilated)
        peaks = peaks & (dist_transform > 1.0)
        
        peaks_uint8 = np.uint8(peaks) * 255
        
        if cv2.countNonZero(peaks_uint8) == 0:
            return opening
        
        ret, markers = cv2.connectedComponents(peaks_uint8)
        markers = markers + 1
        
        sure_bg = cv2.dilate(opening, kernel_small, iterations=3)
        unknown = cv2.subtract(sure_bg, np.uint8(peaks_uint8))
        markers[unknown == 255] = 0
        
        mask_bgr = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(mask_bgr, markers)
        
        mask_out = np.zeros_like(mask)
        mask_out[markers > 1] = 255
        
        return mask_out
    
    
    
    def memory_post(
            self,
            frame_data:tuple
    ):
        """Store frame data to disk for long-term storage"""
        frame_num = frame_data[0]
        frame_path = os.path.join(self.memory_directory, f"frame_{frame_num:05d}.npz")
        np.savez_compressed(frame_path,
                            frame=frame_data[1],
                            fgMask_st=frame_data[2],
                            dbMask=frame_data[3])

    
    def memmory_get(self, frame_idx):
        """Retrieve historical data for back-tracing"""
        frame_path = os.path.join(self.memory_directory, f"frame_{frame_idx:05d}.npz")
        if os.path.exists(frame_path):
            data = np.load(frame_path)
            return {
                'frame': data['frame'],
                'fgMask_st': data['fgMask_st'],
                'dbMask': data['dbMask']
            }

    def memory_delete(self, frame_idx):
        """Remove historical data from disk"""
        frame_path = os.path.join(self.memory_directory, f"frame_{frame_idx:05d}.npz")
        if os.path.exists(frame_path):
            try:
                os.remove(frame_path)
            except OSError as e:
                print(f"Error deleting {frame_path}: {e}")


class IdGenerator:
    def __init__(self):
        self.current_id = 0

    def get_next_id(self):
        self.current_id += 1
        return self.current_id
    

class BlobProcessor:
    def __init__(self, min_area=500, max_area=50000):
        self.min_area = min_area
        self.max_area = max_area
        
        self.hog = cv2.HOGDescriptor()
        
        self.id_gen = IdGenerator()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def process(self, fg_mask, frame, current_frame_idx): 
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_candidates = []

        frame_h, frame_w = frame.shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            
            # --- ZABEZPIECZENIE 
            x = max(0, x)
            y = max(0, y)
            if x + w > frame_w:
                w = frame_w - x
            if y + h > frame_h:
                h = frame_h - y

            if w <= 0 or h <= 0:
                print("Warning: Detected bounding box has non-positive dimensions after clamping. Skipping.")
                continue

            roi = frame[y:y+h, x:x+w]
            
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            v = np.median(gray_roi)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edges = cv2.Canny(gray_roi, lower, upper)
            
            roi_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            max_contour = None
            if roi_contours:
                max_contour = max(roi_contours, key=cv2.contourArea)

            detected_candidates.append(PSO(
                id=self.id_gen.get_next_id(),
                bbox=(x, y, w, h),
                area=area,
                center=(x + w // 2, y + h // 2),
                max_contour=max_contour,
                frame_created_idx=current_frame_idx,
                bg_roi=roi.copy() 
            ))

        return detected_candidates

class AbandonmentDetector:
    def __init__(self):
        self.active_objects = {}

    def update(self, new_candidates, bg_model, frame_img):
        """
        Main logic loop executing steps 1, 3, 4, 5.
        new_candidates: List of PSO objects detected in current frame
        bg_model: Instance of DualBackgroundModel (for memory access)
        current_frame: Frame number
        frame_img: Current frame image (BGR) for CSO verification
        """

        if not self.active_objects:
            self.active_objects = {obj.id: obj for obj in new_candidates}
        else:
            matched_indices = set()
            objects_to_remove = []

            for obj_id, obj in self.active_objects.items():
                best_match = None
                best_iou = 0.2
                
                # --- CSO VERIFICATION (Independent of Foreground Mask) ---
                # Sprawdzamy czy obiekt nadal istnieje w obrazie, nawet jeśli tło go wchłonęło
                if obj.type_name == "CSO":
                    x, y, w, h = obj.bbox
                    fh, fw = frame_img.shape[:2]
                    
                    # 1. Dodajemy margines (padding) do ROI, żeby obiekt nie uciekł przy drobnych ruchach
                    margin = 0
                    if w * h < 80:
                        margin = 20

                    x_roi = max(0, x - margin)
                    y_roi = max(0, y - margin)
                    # Szerokość/wysokość ROI uwzględnia margines z obu stron
                    w_roi = min(fw - x_roi, w + 2 * margin)
                    h_roi = min(fh - y_roi, h + 2 * margin)
                    
                    cso_confirmed = False
                    if w_roi > 0 and h_roi > 0:
                        current_roi = frame_img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
                        gray_roi = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
                        
                        # 2. Zamiast Canny -> Adaptive Threshold + Morfologia
                        # To daje pełniejsze "plamy" (bloby) zamiast cienkich krawędzi
                        mask_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                       cv2.THRESH_BINARY_INV, 11, 2)
                        
                        # Sklejamy dziury w masce
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel, iterations=2)
                        
                        roi_contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Check contour match
                        if roi_contours:
                            # Szukamy największego konturu, który ma sensowny rozmiar (nie szum)
                            max_contour = max(roi_contours, key=cv2.contourArea)
                            
                            if cv2.contourArea(max_contour) > 50:
                                # Zamiast matchShapes, używamy IoU na BoundingBoxach
                                # 1. Wyznaczamy bbox wykrytego konturu wewnątrz ROI
                                xr, yr, wr, hr = cv2.boundingRect(max_contour)
                                
                                # 2. Przeliczamy na współrzędne globalne
                                # x_roi, y_roi to lewy górny róg naszego powiększonego ROI
                                curr_global_bbox = (x_roi + xr, y_roi + yr, wr, hr)
                                
                                # 3. Porównujemy z zapamiętanym bboxem obiektu
                                iou = self._compute_iou(obj.bbox, curr_global_bbox)
                                
                                # print(f"CSO ID {obj.id} IoU check: {iou:.2f}")
                                
                                # Jeśli prostokąty się pokrywają w > 20% (zmniejszony próg bo ROI jest większe), uznajemy obiekt
                                if iou > 0.2:
                                    cso_confirmed = True
                                    # Opcjonalnie: aktualizujemy bbox, żeby śledzić drobne przesunięcia
                                    obj.bbox = curr_global_bbox 
                        
                    if not cso_confirmed:
                        print(f"CSO ID {obj.id} failed authentication (lost visual), removing.")
                        objects_to_remove.append(obj_id)
                        continue # Skip further processing for this object

                # --- PSO VERIFICATION & TRACKING (Using Foreground Mask Candidates) ---
                
                for cand in new_candidates:                    
                    cand_bbox = cand.bbox
                    obj_bbox = obj.bbox

                    iou = self._compute_iou(cand_bbox, obj_bbox)

                    if iou > best_iou:
                        best_iou = iou
                        best_match = cand
                        best_idx = cand.id
                        

                if best_match:
                    matched_indices.add(best_idx)
                    obj.bbox = best_match.bbox 
                    obj.max_contour = best_match.max_contour 
                    if obj.type_name == "PSO":
                        obj.history_centers.append(best_match.center)
                        obj.history_areas.append(best_match.area)
                        obj.history_bboxes.append(best_match.bbox)
                        
                    
                        is_stable = self._check_stability(obj)

                        if is_stable:
                            obj.stability_counter += 1
                        else:
                            obj.stability_counter = 0 
                    
                    
                        if obj.stability_counter >= STABILITY_FRAMES:
                            self.attempt_transition_to_cso(obj, bg_model)
                    

                elif obj.type_name == "PSO":
                    # Jeśli PSO nie ma dopasowania w FG -> znika
                    objects_to_remove.append(obj_id)

            for obj_id in objects_to_remove:
                if obj_id in self.active_objects:
                    self.active_objects.pop(obj_id)

            for cand in new_candidates:
                if cand.id not in matched_indices:
                    self.active_objects[cand.id] = cand

    def attempt_transition_to_cso(self, pso, bg_model):
        """
        Executes Step 3 (Unattended Check) and Step 4 (Presence Authentication)
        """
        new_cso = CSO(pso)
        print(f"PSO ID {pso.id} promoted to CSO ID {new_cso.id}")
        self.active_objects[new_cso.id] = new_cso

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou


    def _check_stability(self, pso):
        """
        Check if the PSO has been stable using IoU.
        """
        if not pso.history_bboxes:
            return True

        # 1. Sprawdzenie nagłych skoków (porównanie z poprzednią klatką)
        last_bbox = pso.history_bboxes[-1]
        if self._compute_iou(pso.bbox, last_bbox) < 0.5:
            return False

        # 2. Sprawdzenie dryfu w czasie (porównanie z oknem 20 klatek)
        # Zamiast sprawdzać całą historię (STABILITY_FRAMES), sprawdzamy lokalną stabilność
        check_window = 20
        lookback_idx = -check_window if len(pso.history_bboxes) >= check_window else 0
        reference_bbox = pso.history_bboxes[lookback_idx]
        
        iou_ref = self._compute_iou(pso.bbox, reference_bbox)
        
        # Wymagamy wysokiego pokrycia (np. 60%), aby uznać obiekt za nieruchomy
        return iou_ref > 0.6
    
    def _pseudo_blob_detector(self, fg_mask):
        """Detect blobs from a foreground mask (grayscale)"""
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_candidates = []

        frame_h, frame_w = fg_mask.shape[:2]
        MIN_AREA = 100
        MAX_AREA = 10000

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            
            # Sanity Checks
            x = max(0, x)
            y = max(0, y)

            if x + w > frame_w:
                w = frame_w - x

            if y + h > frame_h:
                h = frame_h - y

            if w <= 0 or h <= 0:
                continue

            roi = fg_mask[y:y+h, x:x+w]

            # --- Extract contour edges from mask ---
            # fg_mask is already binary, use Canny directly
            edges = cv2.Canny(roi, 50, 150)
            
            roi_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            max_contour = None
            if roi_contours:
                max_contour = max(roi_contours, key=cv2.contourArea)
            

            detected_candidates.append(PSO(
                bbox=(x, y, w, h),
                area=area,
                center=(x + w // 2, y + h // 2),
                max_contour=max_contour
            ))

        return detected_candidates
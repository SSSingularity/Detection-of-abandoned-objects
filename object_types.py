import cv2
import numpy as np

class ObjectType:
    '''Base class for object types'''
    def __init__(
        self, 
        id = None,
        bbox=None, 
        center=None, 
        area=None, 
        max_contour=None, 
        frame_created_idx=0    
    ):
        self.id = id
        self.bbox = bbox            # (x, y, w, h)
        self.center = center        # (cx, cy)
        self.area = area
        self.max_contour = max_contour
        self.frame_created_idx = frame_created_idx # Frame number when object was first detected

class PSO(ObjectType):
    '''Pre-Stationary Object'''
    def __init__(
        self, 
        id = None,
        bbox=None,
        center=None, 
        area=None,
        max_contour=None, 
        frame_created_idx=0, 
        bg_roi=None
    ):
        super().__init__(id, bbox, center, area, max_contour, frame_created_idx)
        self.type_name = "PSO"
        
        # 1. Stability Verification Data
        self.stability_counter = 0     # Number of consecutive stable frames
        self.history_centers = []      # To check position stability
        self.history_areas = []        # To check size stability
        self.history_bboxes = []       # To check IoU stability
        
        # 4. Authentication Data
        self.bg_max_contour = None     # Background contour (for occlusion check)
        self.initial_max_contour = max_contour # Saved at creation
        
        # 5. Template Data
        self.bg_roi = bg_roi           # Image of background before object appeared
        self.owner_histogram = None    # To be filled during back-tracing

class CSO(ObjectType):
    '''Candidate Stationary Object'''
    def __init__(self, pso_obj):
        # Promote PSO to CSO by copying relevant data
        super().__init__(pso_obj.id, pso_obj.bbox, pso_obj.center, pso_obj.area, pso_obj.max_contour)
        self.type_name = "CSO"
        
        # 5. Template Registration
        self.position = pso_obj.bbox
        self.width = pso_obj.bbox[2]
        self.height = pso_obj.bbox[3]
        
        self.max_contour = pso_obj.initial_max_contour
        if pso_obj.max_contour is not None:
            self.initial_max_contour = pso_obj.max_contour
        else:
            self.initial_max_contour = pso_obj.initial_max_contour # Current contour at time of promotion

        if self.initial_max_contour is None:
            print(f"Warning: CSO ID {self.id} has no initial max contour for authentication.")
        self.bg_max_contour = pso_obj.bg_max_contour
        
        self.bg_roi = pso_obj.bg_roi
        self.owner_histogram = pso_obj.owner_histogram

class OCO(ObjectType):
    '''Occluded Object'''
    def __init__(self, bbox=None, center=None, area=None):
        super().__init__(bbox, center, area)
        self.type_name = "OCO"

class ABO(ObjectType):
    '''Abadoned Object'''
    def __init__(self, bbox=None, center=None, area=None):
        super().__init__(bbox, center, area)
        self.type_name = "ABO"  
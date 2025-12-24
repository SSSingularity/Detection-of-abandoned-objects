class ObjectType:
    '''Base class for object types'''
    def __init__(self, bbox=None, center=None, area=None):
        self.bbox = bbox
        self.center = center
        self.area = area

class PSO(ObjectType):
    '''Pre-Stationary Object'''
    def __init__(self, bbox=None, center=None, area=None):
        super().__init__(bbox, center, area)
        self.type_name = "PSO"
        self.wait = 0 

    def check_transition(self, wait_threshold, frame):
        '''Check if the object should transition to CSO'''
        if self.wait < wait_threshold:
            self.wait += 1
        else:
            me = frame[self.bbox[1]:self.bbox[1]+self.bbox[3], self.bbox[0]:self.bbox[0]+self.bbox[2]]


        return 
class CSO(ObjectType):
    '''Candidate Stationary Object'''
    def __init__(self, bbox=None, center=None, area=None):
        super().__init__(bbox, center, area)
        self.type_name = "CSO"

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
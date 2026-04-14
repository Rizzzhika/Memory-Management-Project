class Page:
    def __init__(self, page_id):
        self.size = 4096
        self.page_id = page_id  # page indentifier
        self.current_node = None #the current node of the page
        self.last_accessed_cpu = None #which CPU last accessed the page
        self.access_history = 0b00000000 #tracks access across last 8 scan windows, not just one
        self.accessed_scan = False  #tells if the page was accessed since last scan
        self.achhhcess_frequency = 0 #how many times accessed in current window
        self.lap_level = 0 # LAP level 0 to 8 ( 0=coldest; 8=hottest); computed from access memory
    
    def record_access(self, cpu_id):
        self.last_accessed_cpu = cpu_id
        self.accessed_scan = True
        self.access_frequency += 1
    
    def update_history_on_scan(self):
        # shift left, add 1 if accessed, mask to 8 bits
        if self.accessed_scan:
            self.access_history = ((self.access_history << 1) | 1) & 0xFF
        else:
            self.access_history = (self.access_history << 1) & 0xFF
        
        # reset for next window
        self.accessed_scan = False
        self.access_frequency = 0
        self.old_lap_level = self.lap_level
        self.lap_level = bin(self.access_history).count('1')  # for computing the hotness 
    
    def __repr__(self):
        return (f"Page(id={self.page_id}, "
                f"node={self.current_node.name if self.current_node else None}, "
                f"lap={self.lap_level}, "
                f"history={bin(self.access_history)})")
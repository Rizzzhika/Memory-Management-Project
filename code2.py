from collections import deque

class MemoryNode:
    def __init__(self, node_id, name, tier, capacity, cpu_socket):
        
        self.node_id = node_id  # we can have 0, 1, 2, 3
        self.name = name          # "DRAM_node0", "DCPMM_node2" etc
        self.tier = tier          # "upper" or "lower"
        self.capacity = capacity  # max pages this node holds
        self.cpu_socket = cpu_socket # which CPU socket this node is attached to, -1 means CPU less
        self.pages = deque() # all the pages that are inside this node in a double way queue
        self.lap_lists = {i: [] for i in range(9)} # seperate pages on the basis of their hotness lap level
    
    def get_latency(self, cpu_id):
        is_local = bool(self.cpu_socket == cpu_id)
        if self.tier.upper() == 'UPPER':
            return 100 if is_local else 150
        
        elif self.tier.upper() == 'LOWER':
            return 300 if is_local else 350
        
        else:
            raise ValueError("INVALID TIER !!")
                
    def is_full(self):
        return len(self.pages) >= self.capacity
    
    def free_slots(self):
        return self.capacity - len(self.pages)
    
    def add_page(self, page):
        self.pages.append(page)
        page.current_node = self
        self.lap_lists[page.lap_level].append(page)
    
    def remove_page(self, page):
        self.pages.remove(page)
        if page in self.lap_lists[page.lap_level]:
            self.lap_lists[page.lap_level].remove(page)
        page.current_node = None
    
    def get_least_accessed_page(self): #for finding the least accessed page in the node that is the victim
        for level in range(9):
            if self.lap_lists[level]:
                return self.lap_lists[level][0]
        return None
    
    def utilization(self):
        return len(self.pages) / self.capacity
    
    def update(self, page):
        if page in self.lap_lists[page.old_lap_level]:
            self.lap_lists[page.old_lap_level].remove(page)
        self.add_page(page)

    def __repr__(self):
        return (f"MemoryNode({self.name}, "
                f"{len(self.pages)}/{self.capacity} pages, "
                f"tier={self.tier})")
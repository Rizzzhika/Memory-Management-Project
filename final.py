from collections import deque
import random
scan_threshold = 100
HOT_THRESHOLD = 6
COLD_THRESHOLD = 2
MIGRATION_THRESHOLD = 2
MAX_MIGRATIONS_PER_SCAN = 20

class Page:
    def __init__(self, page_id):
        self.size = 4096
        self.page_id = page_id  # page indentifier
        self.current_node = None #the current node of the page
        self.last_accessed_cpu = None #which CPU last accessed the page
        self.access_history = 0b00000000 #tracks access across last 8 scan windows, not just one
        self.accessed_scan = False  #tells if the page was accessed since last scan
        self.access_frequency = 0 #how many times accessed in current window
        self.lap_level = 0 # LAP level 0 to 8 ( 0=coldest; 8=hottest); computed from access memory
    
    def record_access(self, cpu_id): #this is for the page to record by whom it was accessed 
        self.last_accessed_cpu = cpu_id
        self.accessed_scan = True 
        self.access_frequency += 1 #tells the number of time the page got accessed 
    
    def update_history_on_scan(self):
        # shift left, add 1 if accessed, mask to 8 bits
        if self.accessed_scan:
            self.access_history = ((self.access_history << 1) | 1) & 0xFF 
        else:
            self.access_history = (self.access_history << 1) & 0xFF
        
        # reset for next window
        self.accessed_scan = False
        self.access_frequency = 0
        old_level = self.lap_level
        self.lap_level = self.access_history.bit_count()  # for computing the hotness 
        if self.current_node and old_level != self.lap_level:
            self.current_node.update(self, old_level)
    
    def __repr__(self): #for printing the information
        return (f"Page(id={self.page_id}, "
                f"node={self.current_node.name if self.current_node else None}, "
                f"lap={self.lap_level}, "
                f"history={bin(self.access_history)})")
     

class MemoryNode:
    def __init__(self, node_id, name, tier, capacity, cpu_socket):
        
        self.node_id = node_id  # we can have 0, 1, 2, 3
        self.name = name          # "DRAM_node0", "DCPMM_node2" etc
        self.tier = tier          # "upper" or "lower"
        self.capacity = capacity  # max pages this node holds
        self.cpu_socket = cpu_socket # which CPU socket this node is attached to, -1 means CPU less
        self.pages = deque() # all the pages that are inside this node in a double way queue
        self.lap_lists = {i: [] for i in range(9)} # seperate pages on the basis of their hotness lap level
    
    def get_latency(self, cpu_id): #tells how fast data can be accessed from different memory
        is_local = bool(self.cpu_socket == cpu_id) # if the cpu which is accessing the node is the same which has the node
        if self.tier.upper() == 'UPPER':
            return 100 if is_local else 150
        
        elif self.tier.upper() == 'LOWER':
            return 300 if is_local else 350
        
        else:
            raise ValueError("INVALID TIER !!")
                
    def is_full(self):  #checks if the nodes is full 
        return len(self.pages) >= self.capacity
    
    def free_slots(self): #checks the empty slot in the node
        return self.capacity - len(self.pages)
    
    def add_page(self, page): #add pages to the node
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
    
    def update(self, page, old_level): # when the node is getting scaned then if the hotness changes
        if page in self.lap_lists[old_level]:
            self.lap_lists[old_level].remove(page)
        self.lap_lists[page.lap_level].append(page)

    def __repr__(self):
        return (f"MemoryNode({self.name}, "
                f"{len(self.pages)}/{self.capacity} pages, "
                f"tier={self.tier})")
    
class CPU:
    def __init__(self, cpu_id):
        self.cpu_id = cpu_id
        self.total_accesses = 0  # number of time CPU accesses the memory 
        self.total_latency = 0 # time it took for the cpu to access the data
    
    def access(self, page):
        if page.current_node is None:
            raise ValueError("Page not mapped to any node")
        
        page.record_access(self.cpu_id) # page will get to know it got accessed 
        latency = page.current_node.get_latency(self.cpu_id)  # will get the latency 
        self.total_latency += latency
        self.total_accesses += 1


    def __repr__(self):
        return f"CPU(id={self.cpu_id}, accesses={self.total_accesses})"
    

class MemorySystem:
    def __init__(self):
        self.migration_counter = 0
        # create 4 nodes exactly 
        self.dram_0  = MemoryNode(0,"DRAM_node0", "upper", capacity=20, cpu_socket=0)
        self.dram_1  = MemoryNode(1, "DRAM_node1", "upper", capacity=20, cpu_socket=1)
        self.dcpmm_2 = MemoryNode(2, "DCPMM_node2", "lower", capacity=80, cpu_socket=0)
        self.dcpmm_3 = MemoryNode(3, "DCPMM_node3", "lower", capacity=80, cpu_socket=1)
        
        # 2 CPUs
        self.cpu0 = CPU(0)
        self.cpu1 = CPU(1)
        
        # a hash map, we connect the pages to their page_id for easy access and iteration
        self.page_table = {}
        
        # all nodes in one list for easy iteration
        self.all_nodes = [
            self.dram_0, 
            self.dram_1,
            self.dcpmm_2, 
            self.dcpmm_3
        ]

        # all cpu in one list for easy iteration
        self.all_cpu = [
            self.cpu0,
            self.cpu1
        ]
    
    def initialize_pages(self, num_pages):
        # all pages start in DCPMM — cold start
        # split between two DCPMM nodes
        for i in range(num_pages):
            page = Page(i)
            if i % 2 == 0:
                self.dcpmm_2.add_page(page)
            else:
                self.dcpmm_3.add_page(page)
            self.page_table[i] = page

    def SystemScan (self):
        for node in self.all_nodes:
            for page in (node.pages):
                page.update_history_on_scan()

    def move_logic(self, page, destination):

        if page.current_node == destination:
            return
        home = page.current_node
        home.remove_page(page)
        destination.add_page(page)
        self.migration_counter += 1
    
    def access_page(self, cpu, page):
        cpu.access(page)

    def get_victim(self, node1, node2):
        victim0 = node1.get_least_accessed_page()
        victim1 = node2.get_least_accessed_page()

        if victim0 is None:
            return victim1
        if victim1 is None:
            return victim0

        return victim0 if victim0.lap_level <= victim1.lap_level else victim1
    
    def OPM(self):
    
        for node in [self.dcpmm_2, self.dcpmm_3]:

            # iterate from hottest levels
            for level in range(HOT_THRESHOLD, 9):
                # copy list to avoid modification issues
                for page in list(node.lap_lists[level]):

                    cpu_id = page.last_accessed_cpu

                    # choose local DRAM
                    target_dram = self.dram_0 if cpu_id == 0 else self.dram_1
                    target_dcpmm = node  # fallback

                    # if already in correct place → skip
                    if page.current_node == target_dram:
                        continue

                    if not target_dram.is_full():
                        self.move_logic(page, target_dram)

                    else:
                        victim = self.get_victim(self.dram_0, self.dram_1)

                        if victim and page.lap_level > victim.lap_level:
                            # evict victim to its DCPMM
                            victim_cpu = victim.last_accessed_cpu
                            victim_dcpmm = self.dcpmm_2 if victim_cpu == 0 else self.dcpmm_3

                            self.move_logic(victim, victim_dcpmm)
                            self.move_logic(page, target_dram)
                        # else skip


        for node in [self.dram_0, self.dram_1]:

            for level in range(0, COLD_THRESHOLD + 1):
                for page in list(node.lap_lists[level]):

                    cpu_id = page.last_accessed_cpu
                    target_dcpmm = self.dcpmm_2 if cpu_id == 0 else self.dcpmm_3

                    self.move_logic(page, target_dcpmm)


    def OPMX(self):

        migrations_this_scan = 0

        for node in [self.dcpmm_2, self.dcpmm_3]:

            for level in range(HOT_THRESHOLD, 9):
                for page in list(node.lap_lists[level]):

                    # throttle migrations
                    if migrations_this_scan >= MAX_MIGRATIONS_PER_SCAN:
                        return

                    cpu_id = page.last_accessed_cpu
                    target_dram = self.dram_0 if cpu_id == 0 else self.dram_1

                    # already in correct place
                    if page.current_node == target_dram:
                        continue

                    if (page.access_history & 0b11) != 0b11:
                        continue

                    if not target_dram.is_full():
                        if page.lap_level < (HOT_THRESHOLD + 1):
                            continue
                        else:
                            self.move_logic(page, target_dram)
                            migrations_this_scan += 1

                    else:
                        victim = self.get_victim(self.dram_0, self.dram_1)

                        if victim is None:
                            continue

                        if (page.lap_level - victim.lap_level) < MIGRATION_THRESHOLD:
                            continue  # skip small improvements

                        # evict victim
                        victim_cpu = victim.last_accessed_cpu
                        victim_dcpmm = self.dcpmm_2 if victim_cpu == 0 else self.dcpmm_3

                        self.move_logic(victim, victim_dcpmm)
                        self.move_logic(page, target_dram)

                        migrations_this_scan += 2


        for node in [self.dram_0, self.dram_1]:

            for level in range(0, COLD_THRESHOLD + 1):
                for page in list(node.lap_lists[level]):

                    if migrations_this_scan >= MAX_MIGRATIONS_PER_SCAN:
                        return

                    cpu_id = page.last_accessed_cpu
                    target_dcpmm = self.dcpmm_2 if cpu_id == 0 else self.dcpmm_3

                    if (page.access_history & 0b11) == 0:
                        self.move_logic(page, target_dcpmm)
                        migrations_this_scan += 1


start = MemorySystem()  #we initialized the system that is created the nodes, CPU, etc.

total_pages = 100 # we'll be changing the number of pages to show the simulation in different cases
print("Number of pages:",total_pages)

start.initialize_pages(total_pages)

#simulation loop
step = 0
while(step != 20000):

    number = random.randint(0,99)
    cpu_select = random.choice(start.all_cpu)
    half = total_pages//2


    if number < 80:
        if cpu_select.cpu_id == 0:
            page_id = random.randint(0, half -1 )
    
        else:
            page_id = random.randint(half, total_pages - 1)

    else:
        if cpu_select.cpu_id == 0:
            page_id = random.randint(half, total_pages - 1)
    
        else:
            page_id = random.randint(0, half - 1)

    page_select = start.page_table[page_id] 
    start.access_page(cpu_select,page_select)

    step += 1

    if (step%scan_threshold == 0):
        start.SystemScan()
        start.OPMX()


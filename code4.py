class MemorySystem:
    def __init__(self):
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
    
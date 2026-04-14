class CPU:
    def __init__(self, cpu_id):
        self.cpu_id = cpu_id
        self.total_accesses = 0  # number of time CPU accesses the memory 
        self.total_latency = 0
    
    def access(self, page):
        latency = page.current_node.get_latency(self.cpu_id)
        page.record_access(self.cpu_id)
        self.total_latency += latency
        self.total_accesses += 1


    def __repr__(self):
        return f"CPU(id={self.cpu_id}, accesses={self.total_accesses})"
    

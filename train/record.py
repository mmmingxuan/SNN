class sn():
    def __init__(self):
        # self.norm2 = []
        # self.norm3 = []
        # self.norm4 = []
        # self.norm5 = []
        # self.norm6 = []
        
        # self.avg_norm2 = 0
        # self.avg_norm3 = 0
        # self.avg_norm4 = 0
        # self.avg_norm5 = 0
        # self.avg_norm6 = 0
        
        self.grad_1 = []
        self.grad_before_2 = []
        self.grad_before_3 = []
        self.grad_before_4 = []
        self.grad_after_2 = []
        self.grad_after_3 = []
        self.grad_after_4 = []
        
        self.avg_grad_1 = 0
        self.avg_grad_before_2 = 0
        self.avg_grad_before_3 = 0
        self.avg_grad_before_4 = 0
        self.avg_grad_after_2 = 0
        self.avg_grad_after_3 = 0
        self.avg_grad_after_4 = 0
        
        self.avg_norm = []
        
        self.norms = [[] for _ in range(10)]
        
    def avg(self):
        # self.avg_norm2 = sum(self.norm2) / len(self.norm2) if self.norm2 else 0
        # self.avg_norm3 = sum(self.norm3) / len(self.norm3) if self.norm3 else 0
        # self.avg_norm4 = sum(self.norm4) / len(self.norm4) if self.norm4 else 0
        # self.avg_norm5 = sum(self.norm5) / len(self.norm5) if self.norm5 else 0
        # self.avg_norm6 = sum(self.norm6) / len(self.norm6) if self.norm6 else 0
        
        for i in range(len(self.norms)):
            self.avg_norm.append(sum(self.norms[i]) / len(self.norms[i]))
        
        
        # self.avg_grad_1 = sum(self.grad_1) / len(self.grad_1) if self.grad_1 else 0
        
        # self.avg_grad_before_2 = sum(self.grad_before_2) / len(self.grad_before_2) if self.grad_before_2 else 0
        # self.avg_grad_before_3 = sum(self.grad_before_3) / len(self.grad_before_3) if self.grad_before_3 else 0
        # self.avg_grad_before_4 = sum(self.grad_before_4) / len(self.grad_before_4) if self.grad_before_4 else 0
        
        # self.avg_grad_after_2 = sum(self.grad_after_2) / len(self.grad_after_2) if self.grad_after_2 else 0
        # self.avg_grad_after_3 = sum(self.grad_after_3) / len(self.grad_after_3) if self.grad_after_3 else 0
        # self.avg_grad_after_4 = sum(self.grad_after_4) / len(self.grad_after_4) if self.grad_after_4 else 0

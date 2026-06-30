class BeliefMap:
    def __init__(self,n):
        self.n=n
        self.p=[[1/(n*n) for _ in range(n)] for _ in range(n)]
    def diffuse(self):
        q=[[0]*self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                v=self.p[i][j]/5
                q[i][j]+=v
                if i>0:q[i-1][j]+=v
                if i<self.n-1:q[i+1][j]+=v
                if j>0:q[i][j-1]+=v
                if j<self.n-1:q[i][j+1]+=v
        self.p=q

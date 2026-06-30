import random
class Grid:
    def __init__(self,n,obs):
        self.n=n
        self.obs=set(obs)
    def valid(self,p):
        x,y=p
        return 0<=x<self.n and 0<=y<self.n and p not in self.obs
    def neighbors(self,p):
        x,y=p
        d=[(1,0),(-1,0),(0,1),(0,-1)]
        r=[]
        for dx,dy in d:
            q=(x+dx,y+dy)
            if self.valid(q):
                r.append(q)
        return r
    def random_free(self):
        while True:
            p=(random.randrange(self.n),random.randrange(self.n))
            if p not in self.obs:
                return p

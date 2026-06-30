import random
from planner import astar,h
class Pursuer:
    def __init__(self,pos,r=7):
        self.pos=pos
        self.r=r
    def act(self,w,target):
        if h(self.pos,target)<=self.r:
            p=astar(w,self.pos,target)
            if len(p)>1:self.pos=p[1]
        else:
            n=w.neighbors(self.pos)
            if n:self.pos=random.choice(n)
class Evader:
    def __init__(self,pos):
        self.pos=pos
    def act(self,w,p):
        n=w.neighbors(self.pos)
        if n:self.pos=max(n,key=lambda x:h(x,p))

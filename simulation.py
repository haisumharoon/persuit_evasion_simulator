import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from environment import Grid
from agents import Pursuer,Evader
from belief import BeliefMap
from metrics import Logger
obs={(5,5),(5,6),(5,7),(12,12),(13,12),(14,12),(18,20),(19,20)}
w=Grid(30,obs)
p=Pursuer((2,2))
e=Evader((26,24))
b=BeliefMap(30)
log=Logger()
plt.figure(figsize=(7,7))
for t in range(300):
    b.diffuse()
    p.act(w,e.pos)
    e.act(w,p.pos)
    log.add(t,p.pos,e.pos)
    plt.clf()
    ax=plt.gca()
    ax.set_xlim(0,30)
    ax.set_ylim(0,30)
    ax.set_aspect("equal")
    for o in obs:
        ax.add_patch(Rectangle(o,1,1))
    plt.scatter(p.pos[0]+0.5,p.pos[1]+0.5,s=120)
    plt.scatter(e.pos[0]+0.5,e.pos[1]+0.5,s=120)
    ax.add_patch(plt.Circle((p.pos[0]+0.5,p.pos[1]+0.5),p.r,fill=False))
    plt.pause(0.03)
    if p.pos==e.pos:
        break
log.save("trajectory.csv")
plt.show()

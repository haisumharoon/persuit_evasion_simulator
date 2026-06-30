import csv
class Logger:
    def __init__(self):
        self.rows=[]
    def add(self,t,p,e):
        self.rows.append([t,p[0],p[1],e[0],e[1]])
    def save(self,f):
        with open(f,"w",newline="") as o:
            c=csv.writer(o)
            c.writerow(["t","px","py","ex","ey"])
            c.writerows(self.rows)

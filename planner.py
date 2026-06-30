import heapq
def h(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])
def astar(world,s,g):
    pq=[(0,s)]
    cost={s:0}
    parent={}
    while pq:
        _,u=heapq.heappop(pq)
        if u==g:
            p=[u]
            while u in parent:
                u=parent[u]
                p.append(u)
            return p[::-1]
        for v in world.neighbors(u):
            nc=cost[u]+1
            if v not in cost or nc<cost[v]:
                cost[v]=nc
                parent[v]=u
                heapq.heappush(pq,(nc+h(v,g),v))
    return [s]

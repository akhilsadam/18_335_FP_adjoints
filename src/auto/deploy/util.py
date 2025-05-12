import copy

def update(d1,d2,skip):
    # update d1 with d2, skipping keys in skip
    d3 = {k:v for k,v in d2.items() if k not in skip}
    d1.update(d3)    


def pretty(d, indent=0, out=[]):
    for key, value in d.items():
      out += ('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         out += ('\t' * (indent+1) + str(value))
    return ''.join(out)
    
def internal_update(d, u):
    for k, v in u.items():
        if isinstance(d, list):
            for i in range(len(d)):
                d[i] = internal_update(d[i], u)
        elif k in d.keys():
            d[k] = v
        else:
            for kd in d.keys():
                d[kd] = internal_update(d[kd], u)
    return d
    
def select(d,i, k):
    d2 = copy.deepcopy(d)
    
    dc = d2
    for key in k[:-1]:
        dc = dc[key][0]
    dc = {k[-1]:[dc[k[-1]][i],]} # dc is now the dictionary we want to keep
    
    d2 = internal_update(d2, dc)
    return d2
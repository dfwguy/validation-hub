"""Bounded Slurm bracket expansion for recorded cluster rosters."""
import re

def expand_hostlist(expression,limit=20000):
    groups=re.split(r',(?=[^\]]*(?:\[|$))',expression.strip())
    result=[]
    for group in groups:
        if '[' not in group:
            if group:result.append(group)
            continue
        match=re.fullmatch(r'([^\[]*)\[([^\]]+)\](.*)',group)
        if not match:raise ValueError('Invalid Slurm hostlist')
        prefix,body,suffix=match.groups()
        for item in body.split(','):
            if '-' in item:
                low,high=item.split('-',1)
                if not low.isdigit() or not high.isdigit():raise ValueError('Invalid numeric host range')
                start,end=int(low),int(high)
                if end<start or end-start>limit:raise ValueError('Hostlist range exceeds limit')
                values=(f'{n:0{len(low)}d}' for n in range(start,end+1))
            else:values=[item]
            for value in values:
                result.append(prefix+value+suffix)
                if len(result)>limit:raise ValueError('Hostlist exceeds limit')
    if len(result)>limit:raise ValueError('Hostlist exceeds limit')
    return result

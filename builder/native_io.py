"""Native FR3 readers, including texture-page/model name collisions and empty models."""
import re,struct
import numpy as np
from fr3_models import Reader

def at(raw,start):
    r=Reader(raw,start);size=r.num('Q')
    assert 0<size<256
    name=r.take(size).decode('ascii');assert re.fullmatch(r'[\w+.-]+',name)
    count=r.num('Q');assert 0<=count<100
    effects=[r.effect() for _ in range(count)];tail=r.take(20)
    return {'name':name,'start':start,'end':r.pos,'effects':effects,'tail':tail}

def model(raw,name):
    needle=struct.pack('<Q',len(name))+name.encode();pos=0;found=[]
    while (pos:=raw.find(needle,pos))>=0:
        try:
            m=at(raw,pos)
            if m['effects']:found.append(m)
        except (AssertionError,struct.error,OverflowError,UnicodeDecodeError):pass
        pos+=1
    assert len(found)==1,(name,len(found))
    return found[0]

def buffers(raw,name):
    end=model(raw,name)['end']
    while True:
        try:end=at(raw,end)['end']
        except (AssertionError,struct.error,UnicodeDecodeError):break
    r=Reader(raw,end);indices=np.frombuffer(r.vector(4),dtype='<u4').copy();vstart=r.pos;vertices=r.vector(64)
    assert r.pos==len(raw)-2,(r.pos,len(raw))
    return end,indices,vstart,vertices

def activate():
    import fr3_models,inspect_faces
    fr3_models.model=model;inspect_faces.model=model;inspect_faces.buffers=buffers

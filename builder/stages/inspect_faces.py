"""Read extracted GLB accessors and native FR3 geometry buffers."""
import json,struct
import numpy as np
from fr3_models import Reader,model

def glb(path):
    b=path.read_bytes();n=struct.unpack_from('<I',b,12)[0]
    d=json.loads(b[20:20+n]);bin=b[28+n:]
    def acc(i):
        a=d['accessors'][i];v=d['bufferViews'][a['bufferView']]
        dt=np.dtype({5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a['componentType']])
        k={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        return np.ndarray((a['count'],k),dt,buffer=bin,offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',k*dt.itemsize),dt.itemsize)).copy()
    return d,acc

def buffers(raw,name):
    end=model(raw,name)['end']
    while True:
        size=struct.unpack_from('<Q',raw,end)[0]
        if not 0<size<256:break
        try:
            text=raw[end+8:end+8+size].decode('ascii')
            if any(ord(c)<32 for c in text):break
            m=model(raw,text)
            assert m['start']==end
            end=m['end']
        except (ValueError,AssertionError,UnicodeDecodeError):break
    r=Reader(raw,end);indices=np.frombuffer(r.vector(4),dtype='<u4').copy();vstart=r.pos
    vertices=r.vector(64)
    assert r.pos==len(raw)-2,(r.pos,len(raw))
    return end,indices,vstart,vertices

"""Read the native FR3 texture vector."""
import struct
from fr3_models import Reader

def textures(raw):
    r=Reader(raw,2);r.take(r.num('Q'));out=[]
    for _ in range(r.num('Q')):
        w,h,combo=struct.unpack('<HHI',r.take(8));pixels=r.vector(4)
        name=r.take(r.num('Q')).decode();page=r.take(r.num('Q')).decode();flag=r.take(1)
        out.append(dict(name=name,page=page,width=w,height=h,combo=combo,pixels=pixels,flag=flag))
    return out,r.pos

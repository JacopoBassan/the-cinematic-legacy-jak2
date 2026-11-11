"""Transfer native morph buffers with explicit face and eye adapters."""
import struct
import numpy as np
from scipy.spatial import cKDTree
from fr3_models import Reader,DRAW,model

def closest_surface(points,triangles):
    """Exact closest point amongst 48 nearest triangle centroids, all three edges included."""
    k=min(48,len(triangles));_,cand=cKDTree(triangles.mean(1)).query(points,k=k)
    if k==1:cand=cand[:,None]
    t=triangles[cand];p=points[:,None,:];a=t[:,:,0];u=t[:,:,1]-a;v=t[:,:,2]-a;q=p-a
    dot=lambda x,y:np.sum(x*y,axis=-1)
    uu=dot(u,u);uv=dot(u,v);vv=dot(v,v);qu=dot(q,u);qv=dot(q,v)
    den=uu*vv-uv*uv;safe=np.where(abs(den)>1e-20,den,1)
    b=(qu*vv-qv*uv)/safe;c=(qv*uu-qu*uv)/safe
    bary=np.stack([1-b-c,b,c],-1)
    valid=(np.min(bary,axis=-1)>=0)&(abs(den)>1e-20)
    projected=np.einsum('nki,nkij->nkj',bary,t)
    best=np.where(valid,dot(projected-p,projected-p),np.inf)
    for i,j in [(0,1),(1,2),(2,0)]:
        edge=t[:,:,j]-t[:,:,i];length=dot(edge,edge)
        alpha=np.clip(dot(p-t[:,:,i],edge)/np.maximum(length,1e-20),0,1)
        edgepoint=t[:,:,i]+alpha[:,:,None]*edge;dist=dot(edgepoint-p,edgepoint-p)
        eb=np.zeros_like(bary);eb[:,:,i]=1-alpha;eb[:,:,j]=alpha
        improve=dist<best;bary=np.where(improve[:,:,None],eb,bary);best=np.minimum(best,dist)
    pick=best.argmin(1);row=np.arange(len(points))
    return cand[row,pick],bary[row,pick],np.sqrt(best[row,pick])

def vec(b,size):return struct.pack('<Q',len(b)//size)+b

def draw_list(draws):return struct.pack('<Q',len(draws))+b''.join(draws)

def fix_model(raw,name,high,parts,indices,vertices,*,retarget_fn,eye_fn,eye_slot=0):
    target=model(raw,name);d,a,deformations,count,report=retarget_fn(name,high,parts)
    prims=d['meshes'][0]['primitives'];eyes=eye_fn(high)
    # Imported draws are grouped by envmap. Recover their original primitive IDs
    # from first_index order (the importer appends primitives in source order).
    draws=sorted([x for e in target['effects'] for x in e['draws']],key=lambda x:DRAW.unpack(x)[3])
    assert len(draws)==len(prims)
    byfirst={DRAW.unpack(x)[3]:i for i,x in enumerate(draws)}
    append=[];next_index=len(indices);modified_total=0
    for effect in target['effects']:
        moddraw=[];fixdraw=[];modvertices=[];floatdata=[];intdata=[];newdraws=[]
        for draw in effect['draws']:
            values=list(DRAW.unpack(draw));pi=byfirst[values[3]];p=prims[pi]
            if pi in eyes:values[2]=eye_slot*2+eyes[pi]
            draw=DRAW.pack(*values);newdraws.append(draw)
            if pi not in deformations:
                fixdraw.append(draw);continue
            ids=indices[values[3]:values[3]+values[4]]
            localids=a(p['indices']).ravel();assert len(ids)==len(localids)
            offset=int(ids[0])-int(localids[0]);assert np.array_equal(ids,localids+offset)
            localused=np.unique(localids);dp,dn=deformations[pi]
            if np.max(np.abs(dp[:,localused]))<1e-7:
                fixdraw.append(draw);continue
            used,mapping=np.unique(ids,return_inverse=True)
            modbase=len(modvertices);append.append((mapping+modbase).astype('<u4'))
            values[3]=next_index;next_index+=len(mapping);moddraw.append(DRAW.pack(*values))
            for j,globalidx in enumerate(used):
                vb=vertices[int(globalidx)*64:(int(globalidx)+1)*64];assert len(vb)==64
                modvertices.append(vb);li=int(globalidx)-offset
                # Positions in FR3 use 4096 units per metre, weights use 8192.
                base=np.frombuffer(vb[:32],dtype='<f4').copy()
                exported=a(p['attributes']['POSITION'])[li]*4096
                assert np.max(abs(base[:3]-exported))<0.015,(name,pi,base[:3],exported)
                active=np.flatnonzero((np.linalg.norm(dp[:,li],axis=1)>1e-7)|(np.linalg.norm(dn[:,li],axis=1)>1e-7))
                if not len(active):continue
                floatdata.append(base)
                for ch in active:
                    fd=np.zeros(8,dtype='<f4');fd[:3]=dp[ch,li]*0.5;fd[4:7]=dn[ch,li]/8192
                    floatdata.append(fd);intdata.append(int(ch))
                intdata.extend([0xffffffff,modbase+j]);modified_total+=1
        effect['draws']=newdraws
        if moddraw:
            effect['mod']=draw_list(moddraw)+draw_list(fixdraw)+vec(b''.join(modvertices),64)+vec(b'',2)+vec(b'',1)+struct.pack('<I',0)+vec(np.asarray(floatdata,dtype='<f4').tobytes(),32)+vec(np.asarray(intdata,dtype='<u4').tobytes(),4)
            effect['tail']=effect['tail'][:9]+b'\1'
            assert len(modvertices)<65535
    maxdraw=sum(max(len(e['draws']),len(Reader(e['mod']).draws())) for e in target['effects'])*2
    target['tail']=struct.pack('<I',maxdraw)+target['tail'][4:]
    return target,append,{'model':name,'native_weight_channels':count,'retargeted_vertices':modified_total,'eye_primitive_sides':eyes,'surface_correspondence':report}

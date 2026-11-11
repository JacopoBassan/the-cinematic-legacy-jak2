"""Generate the Jak II cinematic runtime adapter in a private build project."""
import argparse,json
from pathlib import Path

EFFECTS={
    'jakb-lod0':[2,0,4,5,1,6,3],
    'daxter-lod0':[3,0,0,2,3,1,3],
    'sig-lod0':[3,3,0,2,1,3,4],
    'ashelin-lod0':[0,1,3,3,2,2],
    'samos-lod0':[3,2,3,0,2,1,2,4],
    'keira-lod0':[1,4,1,2,3,0],
}
# Reflection intensity follows a native reflective control even where the
# cinematic model introduces a reflection on a formerly plain material.
REFLECTION_EFFECTS={
    **EFFECTS,
    'ashelin-lod0':[0,1,3,3,2,3],
    'samos-lod0':[3,2,3,0,1,1,2,4],
}
IGNORE_ALPHA={
    'sig-lod0':0b0110111,
    'ashelin-lod0':0b101111,
    'samos-lod0':0b00111101,
    'keira-lod0':0b101100,
}

def patch(source):
    s=source
    def replace(old,new):
        nonlocal s
        if s.count(old)!=1:raise ValueError(f'Unsupported source anchor: {old[:80]} ({s.count(old)})')
        s=s.replace(old,new,1)
    predicate='(or '+' '.join('(string= (-> dc mgeo name) "'+n+'")' for n in EFFECTS)+')'
    count='(cond '+' '.join('((string= (-> dc mgeo name) "'+n+'") '+str(len(m))+')' for n,m in EFFECTS.items())+' (else (-> merc-ctrl header effect-count)))'
    replace('(qwc-total 0))','(qwc-total 0) (cinematic-replacement? '+predicate+'))')
    old='''(when (and (zero? (-> (-> *foreground* merc-bucket-info) effect effect-idx merc-path))
                       (zero? (-> (-> *foreground* merc-bucket-info) effect effect-idx disable-draw))
                       (= tex-idx (-> merc-ctrl effect effect-idx texture-index))
                       )'''
    replace(old,'''(when (or cinematic-replacement?
                     (and (zero? (-> (-> *foreground* merc-bucket-info) effect effect-idx merc-path))
                          (zero? (-> (-> *foreground* merc-bucket-info) effect effect-idx disable-draw))
                          (= tex-idx (-> merc-ctrl effect effect-idx texture-index))))''')
    replace('(logior! enable-mask (ash 1 effect-idx))',
            '(when (zero? (-> (-> *foreground* merc-bucket-info) effect effect-idx disable-draw)) (logior! enable-mask (ash 1 effect-idx)))')
    remap=''
    for name,mapping in EFFECTS.items():
        remap+=f'          (when (string= (-> dc mgeo name) "{name}")\n            (let ((original-enable enable-mask) (original-alpha ignore-alpha-mask))\n              (set! enable-mask 0) (set! ignore-alpha-mask 0)\n'
        for dst,src in enumerate(mapping):
            remap+=f'              (when (logtest? original-enable {1<<src}) (logior! enable-mask {1<<dst}))\n              (when (logtest? original-alpha {1<<src}) (logior! ignore-alpha-mask {1<<dst}))\n'
        if IGNORE_ALPHA.get(name,0):
            remap+=f'              (logior! ignore-alpha-mask {IGNORE_ALPHA[name]})\n'
        remap+='            ))\n'
    replace('          ;; flags\n',remap+'          ;; flags\n')
    replace('(set! (-> flags effect-count) (-> merc-ctrl header effect-count))',
            '(set! (-> flags effect-count) '+count+')')
    fade='''            (dotimes (i (-> merc-ctrl header effect-count))
              (set! (-> fades i) (the-as uint (-> (-> *foreground* merc-bucket-info) effect i color-fade)))
              )'''
    ptr='''            (dotimes (i (-> merc-ctrl header effect-count))
              (set! (-> merc-ptrs i) (-> merc-ctrl effect i))
              )'''
    for original,kind in [(fade,'fade'),(ptr,'ptr')]:
        new='            (cond\n'
        for name,mapping in EFFECTS.items():
            new+=f'              ((string= (-> dc mgeo name) "{name}")\n'
            for dst,src in enumerate(REFLECTION_EFFECTS[name] if kind=='fade' else mapping):
                new+=(f'                (set! (-> fades {dst}) (the-as uint (-> (-> *foreground* merc-bucket-info) effect {src} color-fade)))\n' if kind=='fade'
                      else f'                (set! (-> merc-ptrs {dst}) (-> merc-ctrl effect {src}))\n')
            new+='              )\n'
        new+='              (else\n'+original+'\n              ))'
        replace(original,new)
    old='(let ((num-fades (/ (+ (-> merc-ctrl header effect-count) 3) 4)))'
    assert s.count(old)==2
    s=s.replace(old,'(let ((num-fades (/ (+ '+count+' 3) 4)))')
    replace('          ;; end matrix string\n','''          (when cinematic-replacement?
            (dotimes (mat-idx (+ (-> dc mgeo num-joints) 3))
              (when (and (< mat-idx 128) (zero? (-> transferred-matrices mat-idx)))
                (set! (-> transferred-matrices mat-idx) 1)
                (set! (-> matrix-slot-string matrix-out-idx) mat-idx)
                (+! matrix-out-idx 1)
                (set! (-> (the (pointer pointer) dma-buf)) (&+ matrix-buf (* 128 mat-idx)))
                (&+! dma-buf 16) (+! qwc-total 1))))
          ;; end matrix string
''')
    replace('(uses-fp-blerc (and *use-fp-blerc* (pc-merc-blend-shape (the process-drawable (-> dc process)) blerc-weights)))','(uses-fp-blerc #f)')
    anchor='    ;; mark all as unused, until we see a use\n'
    replace(anchor,'    (dotimes (i 40) (set! (-> blerc-weights i) 0.0))\n    (set! uses-fp-blerc (and (or *use-fp-blerc* '+predicate+') (pc-merc-blend-shape (the process-drawable (-> dc process)) blerc-weights)))\n'+anchor)
    anchor='    ;; loop over texture groupe\n'
    replace(anchor,'    (when '+predicate+' (dotimes (i 7) (set! (-> use-flags i) 0)) (set! (-> use-flags 0) 1))\n'+anchor)
    anchor='         ; (format 0 "~D weights: ~X~%" i blerc-weights)'
    replace(anchor,'          (when '+predicate+" (set! vertex-update 'blerc))\n"+anchor)
    return s

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--project',type=Path,required=True)
    a=p.parse_args();relative=Path('goal_src/jak2/engine/gfx/foreground/foreground.gc')
    original=(a.data/relative).read_text();fixed=patch(original)
    (a.project/relative).write_text(fixed)
    from transformations import patch_texture_animation
    texture_relative=Path('goal_src/jak2/engine/gfx/texture/texture-anim.gc')
    (a.project/texture_relative).write_text(patch_texture_animation((a.data/texture_relative).read_text()))
    (a.project.parent/'runtime-effect-map.json').write_text(json.dumps(EFFECTS,indent=2)+'\n')
    print('Generated isolated Jak II foreground and cinematic dark texture adapters')

if __name__=='__main__':main()

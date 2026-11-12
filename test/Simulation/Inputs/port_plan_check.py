import json, sys
j = json.load(open(sys.argv[1]))
eng = j['engines']
print(f"engines {len(eng)} members {sum(len(e['members']) for e in eng)}")
cls = j['classes']
vms = [v for c in cls for v in c['vms']]
print(f"classes {len(cls)} memories {len(vms)} on engines {sum(v['engine'] >= 0 for v in vms)}")
at = {}
for c in cls:
    for v in c['vms']:
        at[c['name'] + ':' + v['name']] = sorted(set(q % j['ii'] for q in c['state']))
ok = sum(sorted(m['r']) == at.get(m['name']) for e in eng for m in e['members'])
print(f"members at their state passes: {ok}")
shared = 0
for e in eng:
    seen = set()
    for m in e['members']:
        shared += len(seen & set(m['r'])); seen |= set(m['r'])
print(f"shared positions: {shared}")

import os
import time
from tempfile import TemporaryDirectory
from collections import namedtuple
from resilience.disk_guard import DiskGuard
NTuple = namedtuple('usage', ['total', 'used', 'free'])

def fake_disk_usage_factory(free_values):
    values=list(free_values); last=free_values[-1]
    def fn(path):
        if values:
            v=values.pop(0)
        else:
            v=last
        return NTuple(total=1000, used=1000-v, free=v)
    return fn

with TemporaryDirectory() as td:
    os.makedirs(os.path.join(td,'activity'),exist_ok=True)
    days=['12-24-25','12-25-25','12-26-25']
    sizes=[100,200,300]
    for d,s in zip(days,sizes):
        with open(os.path.join(td,'activity',f'act_{d}.jsonl'),'wb') as f:
            f.write(b'x'*s)
    fake_fn=fake_disk_usage_factory([1000,500,150])
    dg=DiskGuard(logs_path=td,check_interval=0.1,days=3,disk_usage_fn=fake_fn)
    dg.start()
    time.sleep(0.6)
    dg.stop()
    sysdir=os.path.join(td,'system')
    print('sysdir exists', os.path.isdir(sysdir))
    if os.path.isdir(sysdir):
        for fn in os.listdir(sysdir):
            print('FILE',fn)
            p=os.path.join(sysdir,fn)
            with open(p,'r',encoding='utf-8') as fh:
                for line in fh:
                    print('LINE:',line.strip())
    print('stats',dg.stats())
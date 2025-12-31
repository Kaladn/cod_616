from pathlib import Path
p=Path('.').glob('**/syswarn_*.jsonl')
any=False
for f in p:
    any=True
    b=f.read_bytes()
    print('file',f)
    print('len',len(b))
    print('repr start',repr(b[:600]))
    print('newline count',b.count(b'\n'))
    print('--- lines split ---')
    for i,l in enumerate(b.splitlines()):
        print(i, len(l), repr(l[:200]))
if not any:
    print('no files')
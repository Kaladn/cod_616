from pathlib import Path
p=Path('.')
for f in p.glob('syswarn_*.jsonl'):
    b=f.read_bytes()
    print('file',f)
    print('len',len(b))
    print('newline count',b.count(b'\n'))
    print(repr(b[:500]))
    print('--- split ---')
    for i,l in enumerate(b.splitlines()):
        print(i, len(l), repr(l[:400]))
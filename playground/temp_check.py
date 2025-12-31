from pathlib import Path
p=Path('.').glob('**/syswarn_12-30-25.jsonl')
for f in p:
    b=f.read_bytes()
    print('file',f)
    print('len',len(b))
    print('repr start',repr(b[:400]))
    print('newline count',b.count(b'\n'))
    print('--- lines split ---')
    for i,l in enumerate(b.splitlines()):
        print(i, len(l), repr(l[:200]))
from forge_memory.indexes.hash_index import HashIndex


def test_insert_and_lookup():
    hi = HashIndex()
    hi.insert(('e','t',(8,8),(6,6)), 10)
    hi.insert(('e','t',(8,8),(6,6)), 20)
    res = hi.lookup(('e','t',(8,8),(6,6)))
    assert res == [10,20]


def test_batch_insert():
    hi = HashIndex()
    pairs = [(('e','t', (8,8), (6,6)), i) for i in range(5)]
    hi.batch_insert(pairs)
    res = hi.lookup(('e','t',(8,8),(6,6)))
    assert res == list(range(5))

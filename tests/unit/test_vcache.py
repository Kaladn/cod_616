from forge_memory.indexes.vcache import VCache


def test_vcache_insert_lookup():
    vc = VCache(max_entries=2)
    key1 = ('e','t',(8,8),(6,6))
    key2 = ('e','t2',(8,8),(6,6))
    vc.insert(key1, [10,20])
    assert vc.lookup(key1) == [10,20]
    vc.insert(key2, [30])
    vc.insert(('e','t3',(8,8),(6,6)), [40])
    # key1 should have been evicted since max_entries=2
    assert vc.lookup(key1) is None

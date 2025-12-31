from forge_memory.indexes.bitmap_index import BitmapIndex


def test_set_and_filter():
    bi = BitmapIndex()
    bi.set_bit(0, True)
    bi.set_bit(1, False)
    offsets = [100, 200]
    mapping = {100: 0, 200: 1}
    res = bi.filter_offsets(offsets, mapping)
    assert res == [100]

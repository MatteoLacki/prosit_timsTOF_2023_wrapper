import lmdb
import msgpack
# Reserve 2GB
env = lmdb.open("mydata.lmdb", map_size=2 * 1024 * 1024 * 1024)

with env.begin(write=True) as txn:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")

with env.begin() as txn:
    value = txn.get(b"key1")
    if value:
        print(value.decode())

with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        print(key.decode(), value.decode())



x = 11
x.to_bytes(4, byteorder='little')


# what I want to store? 
# likely call to some 


data = {'id': 1, 'value': 123.45, 'items': [1, 2, 3, 4]}

%%timeit
serialized_data = msgpack.packb(data)

with env.begin(write=True) as txn:
    txn.put(b'key1', serialized_data)


with env.begin(write=False) as txn:
    stored_serialized_data = txn.get(b'key1')

stored_serialized_data == serialized_data
len(serialized_data)

#OK, so idea would be to iterate over the inputs, turn each tuple into sequence of bytes, and retrieve the beginning of the data and perhaps the size to know how far to jump. 
# so format would consist of lmdb + numpy.memmap + msgpack for serialization.

with env.begin(write=False) as txn:
    index, cnt = txn.get(serialized_foo_call)
data[index:index+cnt]
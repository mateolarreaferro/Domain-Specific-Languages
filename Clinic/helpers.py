"""Helper class for reading binary files.
"""

class BinaryFile:
    def __init__(self, path):
        self.file = open(path, "rb")

    def read_bytes(self, first_byte_index, n_bytes):
        self.file.seek(first_byte_index)
        return self.file.read(n_bytes)

    def read_signed_int(self, first_byte_index, n_bytes):
        # https://stackoverflow.com/questions/444591/how-to-convert-a-string-of-bytes-into-an-int
        return int.from_bytes(self.read_bytes(first_byte_index, n_bytes),
                              byteorder="little",
                              signed=True)

    def read_unsigned_int(self, first_byte_index, n_bytes):
        # https://stackoverflow.com/questions/444591/how-to-convert-a-string-of-bytes-into-an-int
        return int.from_bytes(self.read_bytes(first_byte_index, n_bytes),
                              byteorder="little",
                              signed=False)

    def read_ascii_char(self, byte_index):
        return chr(self.read_unsigned_int(byte_index, 1))

if __name__ == "__main__":
    f = BinaryFile("test_cases/fixed_size_fields/test_no_encryption.fsf")

    decryption_key = f.read_signed_int(31, 1)
    print("Decryption key:", decryption_key)
    if decryption_key != 0: raise NotImplementedError

    payload_size = f.read_signed_int(0, 1)
    print("Payload size:", payload_size)
    for i in range(payload_size):
        print(f.read_ascii_char(1 + i))

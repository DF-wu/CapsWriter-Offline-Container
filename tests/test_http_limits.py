import unittest

from util.server.http_limits import UploadTooLargeError, read_upload_limited


class FakeUpload:
    def __init__(self, data: bytes):
        self.data = data
        self.read_sizes = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if not self.data:
            return b""
        chunk = self.data[:size]
        self.data = self.data[size:]
        return chunk


class UploadLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_limit_is_accepted(self):
        upload = FakeUpload(b"abcdef")

        body = await read_upload_limited(upload, max_bytes=6, chunk_size=4)

        self.assertEqual(body, b"abcdef")
        self.assertEqual(upload.data, b"")

    async def test_oversized_upload_fails_before_reading_to_eof(self):
        upload = FakeUpload(b"abcdefghij")

        with self.assertRaises(UploadTooLargeError):
            await read_upload_limited(upload, max_bytes=5, chunk_size=4)

        self.assertEqual(upload.read_sizes, [4, 2])
        self.assertEqual(upload.data, b"ghij")

    async def test_invalid_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            await read_upload_limited(FakeUpload(b"x"), max_bytes=0)


if __name__ == "__main__":
    unittest.main()

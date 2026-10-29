import unittest

from modules.camera import oak_backend


class TestOakBackend(unittest.TestCase):
    """The installed depthai names the backend: 2.x is oak, 3.x is oak4, anything else an error."""

    def test_major_version_names_the_backend(self) -> None:
        self.assertEqual(oak_backend('2.30.0.0'), 'oak')
        self.assertEqual(oak_backend('3.11.0'), 'oak4')

    def test_other_versions_are_an_error(self) -> None:
        with self.assertRaises(ImportError):
            oak_backend('4.0.0')


if __name__ == '__main__':
    unittest.main()

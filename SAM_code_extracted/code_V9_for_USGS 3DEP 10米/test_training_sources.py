from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from training_sources import training_source_snapshot, verify_training_sources


class SourceScopeTests(unittest.TestCase):
    def test_unrelated_additions_ignored_dependency_changes_detected(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'trainV10.py').write_text('import model\n')
            (root/'model.py').write_text('from loss import value\n')
            (root/'loss.py').write_text('value = 1\n')
            expected=training_source_snapshot(root)
            self.assertEqual(set(expected), {'trainV10.py','model.py','loss.py'})
            (root/'preview_v6_land_envelope.py').write_text('unrelated = True\n')
            verify_training_sources(root, expected)
            (root/'loss.py').write_text('value = 2\n')
            with self.assertRaisesRegex(RuntimeError,'loss.py'):
                verify_training_sources(root,expected)
            (root/'loss.py').unlink()
            with self.assertRaisesRegex(RuntimeError,'loss.py'):
                verify_training_sources(root,expected)

    def test_new_import_and_package_configuration_are_checked(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'pkg').mkdir()
            (root/'trainV10.py').write_text('import pkg\n')
            (root/'pkg/__init__.py').write_text('from .model import value\n')
            (root/'pkg/model.py').write_text('value = 1\n')
            (root/'pkg/model.yaml').write_text('width: 4\n')
            expected=training_source_snapshot(root)
            self.assertIn('pkg/model.yaml',expected)
            (root/'pkg/model.yaml').write_text('width: 8\n')
            with self.assertRaisesRegex(RuntimeError,'model.yaml'):
                verify_training_sources(root,expected)


if __name__=='__main__':unittest.main()

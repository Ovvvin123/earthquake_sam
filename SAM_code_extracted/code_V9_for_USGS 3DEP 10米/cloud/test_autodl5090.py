"""Offline checks for environment drift and fail-closed profile selection."""
import json
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

import environment_autodl5090 as env
import manage


class AutoDLTests(unittest.TestCase):
    def lock(self, versions):
        return dict(profile=env.PROFILE, python=platform.python_version(), versions=versions)

    def test_dependency_changes_missing_and_added_rejected(self):
        wanted = {'torch': '2.8.0+cu128', 'numpy': '2.2.6'}
        candidates = [dict(wanted, torch='2.7.1+cu126'), {'torch': wanted['torch']},
                      dict(wanted, triton='3.4.0')]
        for candidate in candidates:
            with self.subTest(candidate=candidate), patch.object(env, 'resolved_versions', return_value=candidate):
                with self.assertRaisesRegex(RuntimeError, 'environment changed'):
                    env.check_lock(self.lock(wanted))

    def test_matching_lock_and_wrong_python(self):
        versions = {'torch': '2.8.0+cu128'}
        with patch.object(env, 'resolved_versions', return_value=versions):
            self.assertEqual(env.check_lock(self.lock(versions)), versions)
            lock = self.lock(versions)
            lock['python'] = '3.10.0'
            with self.assertRaisesRegex(RuntimeError, 'Python mismatch'):
                env.check_lock(lock)

    def test_missing_lock_cannot_bypass_setup(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(env, 'core_check'), patch.object(env.sys, 'prefix', tmp):
            with self.assertRaisesRegex(RuntimeError, 'Missing AutoDL'):
                env.environment_versions()

    def test_unknown_profile_rejected(self):
        with patch.dict(manage.os.environ, {'V9_ENV_PROFILE': 'typo'}):
            with self.assertRaisesRegex(RuntimeError, 'Unknown V9_ENV_PROFILE'):
                manage.environment_check()

    def test_autodl_does_not_skip_vendor_validation(self):
        with patch.dict(manage.os.environ, {'V9_ENV_PROFILE': env.PROFILE}), \
                patch.object(env, 'environment_versions', return_value={}), \
                patch.object(manage, 'verify', side_effect=RuntimeError('corrupt vendor')):
            with self.assertRaisesRegex(RuntimeError, 'corrupt vendor'):
                manage.environment_check()

    def test_invalid_vendor_still_rejected(self):
        import verify_vendor
        self.assertIn('runtime_files', verify_vendor.verify())
        with patch.object(verify_vendor, 'sha256', return_value='bad'):
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                verify_vendor.verify()


if __name__ == '__main__':
    unittest.main()

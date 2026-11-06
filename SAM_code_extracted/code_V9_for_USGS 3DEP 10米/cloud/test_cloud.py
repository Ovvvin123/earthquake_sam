"""Cloud wrapper contracts: never launch a trainer in these tests."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import manage
import verify_vendor


class CloudTests(unittest.TestCase):
    def test_hyperparameters_equal_local_profile(self):
        from run_experiment import read_profile
        local=read_profile(manage.ROOT/'experiments/california_v9_30m_1024_8gb.json')
        cloud=read_profile(manage.ROOT/'experiments/california_v9_30m_1024_cloud.json')
        self.assertEqual(local['batchsize']*local['stepPerMiniBatch'],cloud['batchsize']*cloud['stepPerMiniBatch'])
        for key in ('batchsize','stepPerMiniBatch'):
            local.pop(key);cloud.pop(key)
        for key in ('dataset','dataFile','resDir','backupDir'):
            local.pop(key);cloud.pop(key)
        self.assertEqual(local,cloud)

    def test_defaults_have_no_windows_or_user_home_dependency(self):
        args,checks=manage.settings({})
        self.assertEqual(args['backupDir'],'')
        self.assertEqual(Path(args['dataFile']),manage.ROOT/'data/Earth_dataset_v7/faults.h5')
        self.assertEqual(checks,manage.ROOT/'cloud_checks')

    def test_overrides_and_invalid_names(self):
        args,checks=manage.settings(dict(V9_DATA_DIR='/disk/data',V9_RUNS_DIR='/disk/runs',
            V9_BACKUP_DIR='/other/backup',V9_CHECKS_DIR='/disk/checks',V9_RUN_NAME='trial1'))
        self.assertEqual(args['backupDir'],'/other/backup/trial1')
        self.assertEqual(args['resDir'],'/disk/runs/trial1/')
        for name in ('../bad','a/b','..',''):
            with self.assertRaises(ValueError):manage.settings({'V9_RUN_NAME':name})

    def test_overlapping_paths_rejected(self):
        with self.assertRaises(ValueError):
            manage.settings(dict(V9_DATA_DIR='/disk',V9_RUNS_DIR='/disk/runs'))
        with self.assertRaises(ValueError):
            manage.settings(dict(V9_RUNS_DIR='/disk/runs',V9_CHECKS_DIR='/disk/runs'))

    def test_only_explicit_mode_can_request_execute(self):
        self.assertNotIn('--execute',manage.launcher_command(Path('/check/profile.json'),Path('/check/report')))
        self.assertIn('--execute',manage.launcher_command(Path('/check/profile.json'),Path('/check/report'),True))
        for args in (['train'],['check','--execute']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):manage.main(args)
        result=subprocess.run(['bash',str(manage.ROOT/'cloud/train.sh')],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('No training started',result.stderr)
        result=subprocess.run(['bash',str(manage.ROOT/'cloud/preflight.sh'),'--execute'],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)

    def test_setup_dry_run_makes_no_environment(self):
        with TemporaryDirectory() as tmp:
            dest=Path(tmp)/'new-venv'
            env=dict(os.environ,V9_BOOTSTRAP_PYTHON=sys.executable,V9_VENV=str(dest))
            result=subprocess.run(['bash',str(manage.ROOT/'cloud/setup.sh'),'--dry-run'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('cu126',result.stdout)
            self.assertFalse(dest.exists())
            dest.mkdir()
            result=subprocess.run(['bash',str(manage.ROOT/'cloud/setup.sh')],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,2)

    def test_vendor_checksum_and_corruption_detection(self):
        self.assertTrue(verify_vendor.verify()['installed_source_match'])
        with patch('verify_vendor.sha256',return_value='corrupt'),self.assertRaises(RuntimeError):verify_vendor.verify()

    def test_missing_data_fails_without_fallback(self):
        with TemporaryDirectory() as tmp,self.assertRaises(FileNotFoundError):
            manage.assets_check({'dataFile':str(Path(tmp)/'faults.h5')})

    def test_storage_probe_cleanup(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'persistent'
            self.assertEqual(manage.writable_storage(root)['atomic_rename_test'],'PASS')
            self.assertEqual(list(root.iterdir()),[])


if __name__=='__main__':unittest.main()

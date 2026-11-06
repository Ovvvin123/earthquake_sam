"""Fast CPU contracts for adapters, frozen gradients, teacher and checkpoint loading."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import torch
from torch import nn
from sam2_lora import (QVLoRA, configure_sam, optimizer_groups, training_mode,
                       set_learning_rates, load_run_model, normalized_config,
                       checkpoint_identity, check_checkpoint_identity)
from aio2_ema import EMATeacher, save_ema_checkpoint, load_ema_checkpoint
from training_checkpoint import save_supervised_checkpoint, load_supervised_checkpoint


class Block(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(source, target*3)

    def forward(self, x):
        q, k, v = self.attn.qkv(x).chunk(3, -1)
        return (q+k+v).tanh()


class TinySAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Module()
        trunk = self.image_encoder.trunk = nn.Module()
        trunk.blocks = nn.ModuleList([Block(a, b) for a, b in [(4,4),(4,6),(6,8),(8,10)]])
        trunk.stage_ends = [0, 1, 2, 3]

    def forward(self, x):
        for block in self.image_encoder.trunk.blocks:
            x = block(x)
        return x


class TinyModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.isvit = self.isalign = False
        self.head, self.sam, self.tail = nn.Linear(3,4), TinySAM(), nn.Linear(10,1)
        self.not_sam = nn.ModuleList([self.head, self.tail])
        self.in_sam = nn.ModuleList([self.sam])
        configure_sam(self, config)

    def forward(self, x):
        return self.tail(self.sam(self.head(x)))


def options(mode='lora', **kw):
    return SimpleNamespace(**dict(normalized_config(dict(samAdaptation=mode)),
                                  lr0=1e-3, resDir='same', epochs=3, **kw))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_zero_increment_and_rng_are_exact_even_at_dimension_changes(self):
        model = TinyModel()
        x = torch.randn(2, 5, 3)
        before = model(x).detach()
        rng = torch.get_rng_state().clone()
        configure_sam(model, vars(options()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.testing.assert_close(before, model(x), rtol=0, atol=0)
        layer = model.sam.image_encoder.trunk.blocks[1].attn.qkv
        z = torch.randn(2, 3, 4)
        with torch.no_grad():
            layer.lora_Bq.weight.fill_(.5)
            layer.lora_Bv.weight.fill_(.5)
        torch.testing.assert_close(layer(z).chunk(3,-1)[1], layer.base(z).chunk(3,-1)[1], rtol=0, atol=0)

    def test_frozen_base_and_upstream_gradients_all_modes(self):
        for mode in ('full', 'frozen', 'lora'):
            args = options(mode)
            model = TinyModel(vars(args))
            optimizer = torch.optim.AdamW(optimizer_groups(model, args))
            before = {n:p.detach().clone() for n,p in model.named_parameters()}
            for step in range(3):
                training_mode(model)
                optimizer.zero_grad()
                model(torch.randn(2,4,3)).square().mean().backward()
                self.assertGreater(model.head.weight.grad.norm().item(), 0)
                optimizer.step()
            self.assertFalse(torch.equal(before['head.weight'], model.head.weight))
            for name, parameter in model.sam.named_parameters():
                if mode != 'full' and '.lora_' not in name:
                    self.assertIsNone(parameter.grad)
                    torch.testing.assert_close(before['sam.'+name], parameter, rtol=0, atol=0)
            if mode == 'lora':
                self.assertTrue(any(not torch.equal(before[n], p) for n,p in model.named_parameters() if '.lora_A' in n))
                self.assertTrue(any(not torch.equal(before[n], p) for n,p in model.named_parameters() if '.lora_B' in n))

    def test_stage_selection_modes_and_no_automatic_merge(self):
        cfg = vars(options()); cfg.update(loraStages='last2', loraDropout=.2)
        model = TinyModel(cfg)
        self.assertEqual(len(model.lora_layers), 2)
        training_mode(model)
        layer = model.sam.image_encoder.trunk.blocks[2].attn.qkv
        self.assertTrue(layer.dropout.training)
        base = layer.base.weight.clone()
        with torch.no_grad(): layer.lora_Bq.weight.fill_(1.)
        model.eval(); training_mode(model); model.eval()
        torch.testing.assert_close(base, layer.base.weight, rtol=0, atol=0)
        self.assertFalse(layer.dropout.training)

    def test_named_learning_rates_and_legacy_group_metadata(self):
        for mode in ('full', 'frozen', 'lora'):
            args = options(mode)
            optimizer = torch.optim.AdamW(optimizer_groups(TinyModel(vars(args)), args))
            for group in optimizer.param_groups: group.pop('name'); group.pop('base_lr')
            set_learning_rates(optimizer, args, .125, .25)
            self.assertEqual(optimizer.param_groups[0]['lr'], args.lr0*.125)
            if mode != 'frozen':
                expected = args.loraLr*.125 if mode == 'lora' else args.lr0/8*.25
                self.assertEqual(optimizer.param_groups[1]['lr'], expected)

    def test_teacher_updates_adapters_and_preserves_frozen_base(self):
        model = TinyModel(vars(options()))
        teacher = EMATeacher(model, .75)
        old = {n:p.clone() for n,p in teacher.model.named_parameters()}
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad: p.add_(.4)
        teacher.update(model)
        for n,p in teacher.model.named_parameters():
            expected = old[n]+.1 if dict(model.named_parameters())[n].requires_grad else old[n]
            torch.testing.assert_close(p, expected)
            self.assertFalse(p.requires_grad)

    def test_supervised_resume_preserves_pending_gradients_and_rng(self):
        args = vars(options())
        a = TinyModel(args); opta = torch.optim.AdamW(optimizer_groups(a, options()))
        a(torch.randn(2,3)).square().mean().backward(); opta.step(); opta.zero_grad()
        a(torch.randn(2,3)).square().mean().backward()  # unstepped accumulation
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.pth'
            save_supervised_checkpoint(path,a,opta,2,1,args,{'best': .1})
            x = torch.randn(2,3)
            a(x).square().mean().backward(); opta.step()
            b = TinyModel(args); optb = torch.optim.AdamW(optimizer_groups(b, options()))
            epoch, step, state = load_supervised_checkpoint(path,b,optb,args)
            self.assertEqual((epoch,step,state), (2,1,{'best': .1}))
            y = torch.randn(2,3); torch.testing.assert_close(x,y,rtol=0,atol=0)
            b(y).square().mean().backward(); optb.step()
            for pa,pb in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(pa,pb,rtol=0,atol=0)
            with self.assertRaises(ValueError):
                load_supervised_checkpoint(path,b,optb,dict(args,loraRank=4))

    def test_ema_checkpoint_and_strict_inference_reload(self):
        args = vars(options()); args['emaTeacher'] = True
        model = TinyModel(args); opt = torch.optim.AdamW(optimizer_groups(model, options()))
        teacher = EMATeacher(model)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            save_ema_checkpoint(path/'ema.pth',model,opt,teacher,1,args)
            other = TinyModel(args); otheropt = torch.optim.AdamW(optimizer_groups(other, options()))
            otherteacher = EMATeacher(other)
            self.assertEqual(load_ema_checkpoint(path/'ema.pth',other,otheropt,otherteacher,args),1)
            torch.save(model.state_dict(),path/'model.pth')
            with patch('sam2_lora.build_model', side_effect=TinyModel):
                with self.assertRaises(ValueError): load_run_model(path)
                (path/'model_config.json').write_text(json.dumps(args))
                loaded = load_run_model(path)
                x = torch.randn(2,3)
                torch.testing.assert_close(model(x),loaded(x),rtol=0,atol=0)

    def test_changed_data_fingerprint_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'key_train.lst'; path.write_text('a')
            cfg = dict(dataset=str(path),samAdaptation='lora')
            ck = dict(identity=checkpoint_identity(cfg))
            path.write_text('b changed')
            with self.assertRaises(ValueError): check_checkpoint_identity(ck,cfg)

    def test_standalone_mirrors(self):
        root = Path(__file__).parent
        self.assertEqual((root/'sam2_lora.py').read_bytes(),
                         (root/'V5_different_split/sam2_lora.py').read_bytes())


if __name__ == '__main__':
    unittest.main()

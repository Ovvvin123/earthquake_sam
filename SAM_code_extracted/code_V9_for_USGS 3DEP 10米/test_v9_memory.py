"""Output/gradient/state compatibility checks; no formal training."""
from copy import deepcopy
import unittest
import torch
from torch import nn
from torchUnet import AttnBlock, conv_sam
from training_memory import recompute
from sam2_lora import normalized_config


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_attention_math_output_and_gradients(self):
        old = AttnBlock(32,head_num=2)
        # Nontrivial output projection: do not hide errors behind near-zero init.
        with torch.no_grad():
            old.proj0.weight.normal_(0,.05); old.proj1.weight.normal_(0,.05)
        new = deepcopy(old); new.attention_backend='sdpa'
        x = torch.randn(2,32,7,9,requires_grad=True)
        z = x.detach().clone().requires_grad_()
        a,b = old(x),new(z)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        a.square().mean().backward(); b.square().mean().backward()
        torch.testing.assert_close(x.grad,z.grad,rtol=1e-4,atol=1e-7)
        for p,q in zip(old.parameters(),new.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-7)
        self.assertEqual(list(old.state_dict()),list(new.state_dict()))

    def test_pool_first_preserves_affine_output_and_gradients(self):
        old = nn.Sequential(nn.Conv2d(8,16,1),nn.AvgPool2d(4,4)).double()
        new = deepcopy(old)
        holder = nn.Module(); holder.conv0=new; holder.memory_options={'poolFirst':True}
        x = torch.randn(2,8,32,32,dtype=torch.float64,requires_grad=True)
        z = x.detach().clone().requires_grad_()
        a,b = old(x),conv_sam._skip(holder,0,z)
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
        a.square().mean().backward(); b.square().mean().backward()
        torch.testing.assert_close(x.grad,z.grad,rtol=1e-9,atol=1e-11)
        for p,q in zip(old.parameters(),new.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-9,atol=1e-11)

    def test_checkpoint_trainable_weights_without_input_grad(self):
        a=nn.Sequential(nn.Linear(4,8),nn.Dropout(.25),nn.Linear(8,3))
        b=deepcopy(a); x=torch.randn(3,4)
        state=torch.get_rng_state()
        y=recompute(a,x,enabled=False);y.square().sum().backward()
        torch.set_rng_state(state)
        z=recompute(b,x,enabled=True);z.square().sum().backward()
        torch.testing.assert_close(y,z)
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p.grad,q.grad)

    def test_legacy_defaults(self):
        cfg=normalized_config({})
        self.assertEqual(cfg['computePrecision'],'fp32')
        self.assertEqual(cfg['attentionBackend'],'legacy')
        self.assertFalse(cfg['poolFirst'])
        self.assertEqual(cfg['activationCheckpoint'],'none')

    def test_execution_settings_resume_roundtrip_and_mismatch(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from test_sam2_lora import TinyModel,options
        from aio2_ema import EMATeacher,save_ema_checkpoint,load_ema_checkpoint
        cfg=vars(options(computePrecision='bf16',attentionBackend='flash',poolFirst=True,
                         activationCheckpoint='decoder',emaTeacher=True))
        a=TinyModel(cfg);oa=torch.optim.AdamW(a.parameters());ta=EMATeacher(a)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'temporary.pth'
            save_ema_checkpoint(path,a,oa,ta,350,cfg)
            b=TinyModel(cfg);ob=torch.optim.AdamW(b.parameters());tb=EMATeacher(b)
            self.assertEqual(load_ema_checkpoint(path,b,ob,tb,cfg),350)
            with self.assertRaisesRegex(ValueError,'computePrecision'):
                load_ema_checkpoint(path,b,ob,tb,dict(cfg,computePrecision='fp32'))


if __name__=='__main__': unittest.main()

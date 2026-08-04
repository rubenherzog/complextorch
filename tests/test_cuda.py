import pytest
import torch
from complextorch.measures import compute_cmem
from complextorch.representations import build_var_system
from complextorch.simulate import random_stable_var,simulate_var
from complextorch.var import VAR

@pytest.mark.cuda
def test_cuda_cpu_parity():
    if not torch.cuda.is_available(): pytest.skip('CUDA is not available')
    coef,q=random_stable_var(16,5,3,seed=45); x=simulate_var(coef,q,500,burnin=500,seed=46)
    cpu=VAR(order=3,fit_intercept=False,device='cpu',stability='ignore').fit(x)
    gpu=VAR(order=3,fit_intercept=False,device='cuda',stability='ignore').fit(x)
    torch.testing.assert_close(cpu.coef_,gpu.coef_.cpu(),rtol=1e-8,atol=1e-8)
    cpu_m=compute_cmem(build_var_system(coef,q),5); gpu_m=compute_cmem(build_var_system(coef.cuda(),q.cuda()),5)
    torch.testing.assert_close(cpu_m.cmem3_curve,gpu_m.cmem3_curve.cpu(),rtol=1e-8,atol=1e-8)

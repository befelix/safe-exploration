# content of conftest.py


try:
    import safe_exploration.ssm_gpy
    safe_exploration.ssm_gpy.__name__ #Need this, otherwise we get an unused import error
    _has_ssm_gpy_ = True
except:
    _has_ssm_gpy_ = False

try:
    import safe_exploration.ssm_pytorch
    safe_exploration.ssm_pytorch.__name__ #Need this, otherwise we get an unused import error
    _has_ssm_pytorch_ = True
except:
    _has_ssm_pytorch_ = False

collect_ignore = ["setup.py"]
if not _has_ssm_gpy_:
    collect_ignore.append("safe_exploration/ssm_gpy")
if not _has_ssm_pytorch_:
    collect_ignore.append("safe_exploration/ssm_pytorch")

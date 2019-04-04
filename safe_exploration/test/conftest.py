# content of conftest.py
import pytest

try:
    import safe_exploration.ssm_gpy
    safe_exploration.ssm_gpy.__name__
    _has_ssm_gpy_ = True
except:
    _has_ssm_gpy_ = False

try:
    import safe_exploration.ssm_pytorch
    safe_exploration.ssm_pytorch.__name__
    _has_ssm_pytorch = True
except:
    _has_ssm_pytorch = False

collect_ignore = ["setup.py"]
if not _has_ssm_gpy_:
    collect_ignore.append("safe_exploration/ssm_gpy")
if not _has_ssm_pytorch:
    collect_ignore.append("safe_exploration/ssm_pytorch")


@pytest.fixture(scope="session")
def check_has_ssm_pytorch():
    if not _has_ssm_pytorch:
        pytest.skip("Optional package 'ssm_pytorch' required to run this test")


@pytest.fixture(scope="session")
def check_has_ssm_gpy():
    if not _has_ssm_gpy_:
        pytest.skip("Optional package 'ssm_gpy' required to run this test")

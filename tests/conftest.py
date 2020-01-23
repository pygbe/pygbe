import pytest

def pytest_addoption(parser):
    parser.addoption('--arch', action = 'store', default = 'gpu', help = 'You can use --arch=gpu or --arch=gpu. By default PyGBe will use GPU.')

@pytest.fixture
def arch(request):
    return request.config.getoption('--arch')

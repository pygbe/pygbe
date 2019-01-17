import pytest

def pytest_addoption(parser):
    parser.addoption('--arc', action = 'store', default = 'gpu', help = 'You can use --arc=gpu or --arc=gpu.')

@pytest.fixture
def arc(request):
    return request.config.getoption('--arc')

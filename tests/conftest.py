import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--models",
        action="store_true",
        default=False,
        help="run tests that download the real models (also RUACCENT_TEST_MODELS=1)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "models: needs the real ONNX models and network access")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--models") or os.environ.get("RUACCENT_TEST_MODELS") == "1":
        return
    skip = pytest.mark.skip(reason="pass --models or set RUACCENT_TEST_MODELS=1")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)

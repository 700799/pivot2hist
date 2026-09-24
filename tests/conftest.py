import pandas as pd
import pytest

import pivot2hist as p2h


@pytest.fixture(scope="session")
def fw() -> pd.DataFrame:
    return p2h.sample.firewall_logs(3000, seed=0)


@pytest.fixture(scope="session")
def auth() -> pd.DataFrame:
    return p2h.sample.auth_logs(1500, seed=1)

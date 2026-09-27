import pandas as pd
import pytest

import bts_pivot as bp


@pytest.fixture(scope="session")
def fw() -> pd.DataFrame:
    return bp.sample.firewall_logs(3000, seed=0)


@pytest.fixture(scope="session")
def auth() -> pd.DataFrame:
    return bp.sample.auth_logs(1500, seed=1)

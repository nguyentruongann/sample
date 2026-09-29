from __future__ import annotations

import sys

from demand_forecasting.pipeline import main

if __name__ == "__main__":
    main(["ingest-live", *sys.argv[1:]])

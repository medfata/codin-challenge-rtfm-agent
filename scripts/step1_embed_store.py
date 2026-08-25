"""Step 1 CLI: run the ingestion pipeline (load -> chunk -> embed -> store in Redis)."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from redis import Redis

from rtfm_agent.config import REDIS_URL
from rtfm_agent.ingest import run_ingestion
from rtfm_agent.tenancy import TenantContext, normalize_tenant


def main():
    parser = argparse.ArgumentParser(description="Step 1: ingest docs into Redis")
    parser.add_argument("--tenant", default="local", help="tenant id to ingest for (default: local)")
    args = parser.parse_args()

    ctx = normalize_tenant(args.tenant)
    if ctx is None:
        print(f"Error: invalid or disallowed tenant '{args.tenant}'")
        sys.exit(1)
    ctx = TenantContext(ctx.id)

    print("=" * 60)
    print(f"Step 1: Document Ingestion Pipeline (tenant={ctx.id})")
    print("=" * 60)

    r = Redis.from_url(REDIS_URL, decode_responses=False)
    r.ping()
    print(f"Connected to {REDIS_URL}\n")

    summary = run_ingestion(r, ctx, verbose=True)

    print("\n" + "=" * 60)
    print("Step 1 Complete!")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()

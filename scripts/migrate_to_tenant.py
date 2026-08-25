"""One-shot migration of legacy single-tenant Redis data into a tenant namespace.

Renames every unprefixed legacy key onto the target tenant prefix:
    doc:{file}:{pos}     -> t:<target>:doc:{file}:{pos}
    cache:{uuid}         -> t:<target>:cache:{uuid}
    session:{sid}:*      -> t:<target>:session:{sid}:*
    metrics:*            -> t:<target>:metrics:*

Pattern-safety assumption (verified): the legacy scan patterns ("doc:*",
"cache:*", "session:*", "metrics:*") can never match an already-migrated key -
prefixed keys start with the literal "t:" while every pattern starts with the
surface name ("doc:" etc.), so the two glob sets are disjoint. Re-running the
migration therefore finds nothing to move and is idempotent.

After renaming, the tenant's doc + cache FT indexes are recreated so the
renamed hashes are re-indexed under t:<target>:- RediSearch binds indexes to
key prefixes, so a RENAME alone would leave hashes invisible to retrieval.
The old unprefixed "doc_idx"/"cache_idx" index names never match the scan
patterns (no colon), so they are left untouched. Use --dry-run to preview.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from redis import Redis

from rtfm_agent import cache as cache_mod
from rtfm_agent import ingest as ingest_mod
from rtfm_agent.config import EMBEDDING_DIM, REDIS_URL
from rtfm_agent.tenancy import normalize_tenant

LEGACY_PATTERNS = ("doc:*", "cache:*", "session:*", "metrics:*")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate legacy unprefixed Redis keys into a tenant namespace"
    )
    parser.add_argument("--target", default="default",
                        help="target tenant slug (default: default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would move without writing anything")
    args = parser.parse_args()

    ctx = normalize_tenant(args.target)
    if ctx is None:
        print(f"Error: invalid or disallowed target tenant '{args.target}'")
        sys.exit(1)

    r = Redis.from_url(REDIS_URL, decode_responses=False)
    try:
        r.ping()
    except Exception as exc:
        print(f"Error: Redis unreachable at {REDIS_URL}: {str(exc)[:200]}")
        sys.exit(1)

    print("=" * 60)
    print(f"Migrating legacy keys -> tenant '{ctx.id}' "
          f"(prefix '{ctx.prefix}')" + ("  [DRY RUN]" if args.dry_run else ""))
    print(f"Redis: {REDIS_URL}")
    print("=" * 60)

    moved = {pattern: 0 for pattern in LEGACY_PATTERNS}
    skipped = {pattern: 0 for pattern in LEGACY_PATTERNS}

    for pattern in LEGACY_PATTERNS:
        for raw_key in r.scan_iter(match=pattern):
            key = raw_key.decode(errors="replace") if isinstance(raw_key, bytes) else str(raw_key)
            new_name = f"{ctx.prefix}{key}"
            if r.exists(new_name):
                skipped[pattern] += 1
                print(f"  [SKIP ] {key} -> {new_name} (target already exists)")
                continue
            if args.dry_run:
                moved[pattern] += 1
                print(f"  [MOVE?] {key} -> {new_name}")
                continue
            try:
                r.rename(key, new_name)
            except Exception as exc:
                print(f"  [WARN ] rename failed for {key}: {str(exc)[:150]}")
                continue
            moved[pattern] += 1
            print(f"  [MOVED] {key} -> {new_name}")

    total_moved = sum(moved.values())
    total_skipped = sum(skipped.values())

    doc_index_created = None
    cache_index_created = None
    if args.dry_run:
        print("\nDry run complete - no keys renamed, no indexes touched.")
    else:
        doc_index_created = ingest_mod.create_redis_index(r, ctx, EMBEDDING_DIM)
        cache_index_created = cache_mod.ensure_cache_index(r, ctx)

    print("\nSummary")
    print("-" * 60)
    for pattern in LEGACY_PATTERNS:
        label = f"{moved[pattern]} moved, {skipped[pattern]} skipped"
        print(f"  {pattern:<12} {label}")
    print(f"  {'TOTAL':<12} {total_moved} moved, {total_skipped} skipped")
    if args.dry_run:
        print("  indexes: not touched (dry run)")
    else:
        print(f"  doc index   '{ctx.doc_index}' recreated: {doc_index_created}")
        print(f"  cache index '{ctx.cache_index}' recreated: {cache_index_created}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Multi-Tenancy Plan (Step 8)

Scope every Redis key, FT index, and Agent Memory Server namespace by tenant; identity is one required header, not an auth system.

## Identity & validation

| Rule | Behavior |
|---|---|
| Required header | `X-Tenant-Id` on every request except `/health` |
| Slug rule | `^[a-z0-9][a-z0-9-_]{0,62}$`, lowercased before matching - no `:` or wildcard characters can reach the Redis key space (blocks key-space injection) |
| Missing / malformed | `422` with the expected-slug hint |
| Valid slug, not allowlisted | `403` naming the rejected tenant |
| Allowlist | env `TENANTS`: comma-separated slugs (`acme,globex`) or `*` (open mode - any valid slug accepted) |
| Wiring | `TENANTS_OPEN` / `TENANT_ALLOWLIST` in `rtfm_agent/config.py`; validation + FastAPI dependency in NEW module `rtfm_agent/tenancy.py` |

A validated id becomes a `TenantContext` deriving everything downstream: `.prefix` (`t:{org}:`), `.doc_index`, `.cache_index`, `.metrics_key`.

## Key layout

| Surface | Single-tenant | Multi-tenant |
|---|---|---|
| Doc chunk hash | `doc:{file}:{pos}` | `t:{org}:doc:{file}:{pos}` |
| Doc FT index | `doc_idx` | `t:{org}:doc_idx` |
| Cache entry hash | `cache:{uuid}` | `t:{org}:cache:{uuid}` |
| Cache FT index | `cache_idx` | `t:{org}:cache_idx` |
| Session keys | `session:{sid}:msgs` / `:summary` | `t:{org}:session:{sid}:msgs` / `t:{org}:session:{sid}:summary` |
| Metrics hash | one shared counter hash | `t:{org}:metrics:cache` (per-tenant only; no shared/global view) |

Per-tenant doc/cache indexes are created lazily via self-healing (missing or wrong-dim index is recreated once and retried), so a brand-new tenant needs no setup step.

## Memory namespacing

The Agent Memory Server has no Redis key prefixes, so isolation rides on namespaces instead: working-memory PUTs send `namespace={org}`, and every long-term search filters `{"namespace": {"eq": org}}`. Isolation-first by design - if the server rejects the filter, search returns `[]` rather than falling back to an unfiltered query that could surface another tenant's facts.

## Per-team ingestion

`POST /ingest` accepts an optional body `{"docs_dir": "..."}`; source directory resolution precedence:

1. `docs/<tenant>/` when it exists
2. validated override (must resolve to a directory inside the project root)
3. `DOCS_DIR` default

CLI parity: `scripts/step1_embed_store.py` gained `--tenant` (default `local`). `scripts/migrate_to_tenant.py` renames legacy unprefixed keys into `t:<target>:` (default target `default`) and recreates the indexes; `--dry-run` supported.

## Touchpoints

| File | Change |
|---|---|
| `rtfm_agent/tenancy.py` (new) | Slug regex + allowlist check + `require_tenant` dependency + `TenantContext` naming derivation |
| `rtfm_agent/config.py` | `TENANTS` parsing -> `TENANTS_OPEN` / `TENANT_ALLOWLIST` |
| `api.py` | `require_tenant` on all routes except `/health`; optional `{docs_dir}` body on `/ingest` |
| `ingest.py` / `retrieval.py` / `scope.py` | Keys + index lookups via `TenantContext`; lazy self-healing per-tenant doc index |
| `cache.py` | Tenant-scoped entries/index; flush + count limited to the tenant's prefix |
| `sessions.py` | All session keys under `t:{org}:session:` |
| `metrics.py` | Counters moved to per-tenant hash `t:{org}:metrics:cache` |
| `memory.py` | `namespace={org}` on PUT; eq-filter on search; fail-closed to `[]` |
| `scripts/step1_embed_store.py` | `--tenant` flag (default `local`) |
| `scripts/step8_check_multitenancy.py` (new) | Isolation drills, no LLM |
| `scripts/migrate_to_tenant.py` (new) | One-shot legacy-key migration |

## Test plan

1. HTTP probes: `/health` reachable without header; all other endpoints 422 without/malformed header, 403 for off-list slugs, 200 with valid ones
2. Doc drill: tenant A ingests, tenant B retrieves nothing from A's corpus (and vice versa); separate `doc_idx` listings per tenant
3. Cache drill: answer cached for A never served to B; flush removes only the calling tenant's entries
4. Session drill: B cannot read A's history even guessing the session id
5. Metrics drill: counters independent per tenant; no global aggregation exposed
6. Memory namespace drill: fact saved as A not recallable as B; filter-rejection path returns `[]`
7. Key-hygiene scan: zero keys outside `t:<slug>:` space after all drills
8. Migration drill: `migrate_to_tenant.py --dry-run` then real run renames legacy keys into `t:default:` and recreates indexes

All step 8 checks run in `scripts/step8_check_multitenancy.py` without any LLM calls.

## Tradeoffs

- No authentication: `X-Tenant-Id` is trusted identity at the same trust level as the rest of the deployment; anyone who can reach the API can act as any allowed tenant
- No per-tenant rate limits or quotas - all tenants share the same LLM free-tier pools
- Strict mode breaks clients that do not send the header (existing ones were updated)
- 2 extra FT indexes per tenant - fine for tens of teams; at hundreds, reconsider a TAG-filter strategy over one shared index

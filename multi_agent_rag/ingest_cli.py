"""
Data ingestion & maintenance CLI.

Usage (run from multi_agent_rag/ or via scripts/ingest.sh):

  python ingest_cli.py list-datasets
  python ingest_cli.py dataset --name wikipedia-simple --max-docs 500
  python ingest_cli.py dataset --hf-path cnn_dailymail --hf-config 3.0.0 \
                               --text-field article --max-docs 200
  python ingest_cli.py refresh              # re-pull all previously ingested datasets
  python ingest_cli.py sync                 # rebuild the vector index incrementally
  python ingest_cli.py sync --force         # full re-embed (after config changes)
  python ingest_cli.py check                # verify LLM provider + index health
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_logger  # noqa: E402

log = get_logger("IngestCLI")


def cmd_list_datasets(_args) -> int:
    from rag.dataset_loader import list_presets
    print("\nAvailable open-source dataset presets:\n")
    for name, desc in list_presets().items():
        print(f"  {name:24s} {desc}")
    print("\nAny other Hugging Face dataset works too:\n"
          "  python ingest_cli.py dataset --hf-path <path> --text-field <field>\n")
    return 0


def cmd_dataset(args) -> int:
    from rag.dataset_loader import load_preset, load_hf_dataset
    if args.hf_path:
        stats = load_hf_dataset(
            path=args.hf_path, config_name=args.hf_config, split=args.split,
            text_field=args.text_field, title_field=args.title_field,
            max_docs=args.max_docs,
        )
    elif args.name:
        stats = load_preset(args.name, max_docs=args.max_docs)
    else:
        log.error("Provide --name <preset> or --hf-path <hf dataset path>")
        return 2
    print(json.dumps(stats, indent=2))
    if not args.no_sync:
        return cmd_sync(args)
    return 0


def cmd_refresh(args) -> int:
    from rag.dataset_loader import refresh_all
    results = refresh_all(max_docs=args.max_docs)
    print(json.dumps(results, indent=2))
    failed = [r for r in results if "error" in r]
    if not args.no_sync:
        rc = cmd_sync(args)
        return rc or (1 if failed else 0)
    return 1 if failed else 0


def cmd_sync(args) -> int:
    from rag import IndexManager
    force = getattr(args, "force", False)
    report = IndexManager().sync(force=force)
    print(json.dumps(report, indent=2))
    return 0


def cmd_check(_args) -> int:
    from llm import check_provider
    from rag import IndexManager

    status = check_provider()
    print(f"\nLLM provider : {status['provider']} (model={status['model']})")
    print(f"  status     : {'OK' if status['ok'] else 'FAILED'} — {status['detail']}")

    mgr = IndexManager()
    report = mgr.sync()
    print(f"\nKnowledge base: {report['documents']} documents, "
          f"{report['chunks']} chunks ({report['backend']} backend)")
    print(f"  sync time  : {report['seconds']}s "
          f"({report['embedded_cached']} cached / {report['embedded_new']} new embeddings)\n")
    return 0 if status["ok"] and report["chunks"] > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Agent RAG data management")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-datasets", help="Show available open-source dataset presets")

    p_ds = sub.add_parser("dataset", help="Ingest an open-source dataset")
    p_ds.add_argument("--name", help="Preset name (see list-datasets)")
    p_ds.add_argument("--hf-path", help="Any Hugging Face dataset path")
    p_ds.add_argument("--hf-config", default=None, help="HF dataset config name")
    p_ds.add_argument("--split", default="train")
    p_ds.add_argument("--text-field", default="text")
    p_ds.add_argument("--title-field", default=None)
    p_ds.add_argument("--max-docs", type=int, default=None)
    p_ds.add_argument("--no-sync", action="store_true", help="Skip index rebuild")

    p_rf = sub.add_parser("refresh", help="Re-pull all previously ingested datasets (for cron)")
    p_rf.add_argument("--max-docs", type=int, default=None)
    p_rf.add_argument("--no-sync", action="store_true")

    p_sync = sub.add_parser("sync", help="Incrementally rebuild the vector index")
    p_sync.add_argument("--force", action="store_true", help="Ignore embedding cache")

    sub.add_parser("check", help="Health-check the LLM provider and the index")

    args = parser.parse_args()
    if getattr(args, "max_docs", None) is None and hasattr(args, "max_docs"):
        from config import DATASET_MAX_DOCS
        args.max_docs = DATASET_MAX_DOCS

    handlers = {
        "list-datasets": cmd_list_datasets,
        "dataset": cmd_dataset,
        "refresh": cmd_refresh,
        "sync": cmd_sync,
        "check": cmd_check,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        return 130
    except Exception as e:  # noqa: BLE001 — CLI boundary: report, don't traceback-spam
        log.error("Command failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

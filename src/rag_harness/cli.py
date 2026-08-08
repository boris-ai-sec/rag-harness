"""Operator CLI for bounded run-native Harness execution."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag_harness.contracts import RunConfiguration, RunPackage
from rag_harness.corpus import CONTROLLED_CORPUS
from rag_harness.governed_export import export_verified_run_v03
from rag_harness.governed_reuse import reuse_synthetic_export_v03
from rag_harness.registry import get_experiment
from rag_harness.runner import execute_experiment
from rag_harness.verification import verify_run_package

EXIT_PASS = 0
EXIT_EXPERIMENT_FAIL = 1
EXIT_INPUT_ERROR = 2
EXIT_BLOCKED = 3
EXIT_EXECUTION_FAILED = 4
EXIT_VERIFICATION_FAILED = 5
EXIT_GOVERNED_EXPORT_FAILED = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-harness",
        description="Execute controlled run-native RAG Harness experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run",
        help="execute one experiment configuration",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to a run-native JSON configuration",
    )
    run_parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="directory in which the non-overwriting run workspace is created",
    )
    run_parser.add_argument(
        "--run-id",
        help="optional explicit run UUID in run-<uuid> form",
    )
    identity_group = run_parser.add_mutually_exclusive_group()
    identity_group.add_argument(
        "--case-id",
        help="override the configuration for a client-scoped case execution",
    )
    identity_group.add_argument(
        "--standalone",
        action="store_true",
        help="clear case_id for a standalone run-native laboratory execution",
    )
    verify_parser = subparsers.add_parser(
        "verify",
        help="verify one finalized run package without modifying it",
    )
    verify_parser.add_argument(
        "--package",
        type=Path,
        required=True,
        help="path to the authoritative run_package.json",
    )
    export_parser = subparsers.add_parser(
        "export-v03",
        help="export a verified run into governed V0.3 Layer 2 evidence",
    )
    export_parser.add_argument(
        "--package",
        type=Path,
        required=True,
        help="path to the authoritative run_package.json",
    )
    export_parser.add_argument(
        "--framework-root",
        type=Path,
        required=True,
        help="path to the extracted frozen Technical Prototype V0.3 root",
    )
    export_parser.add_argument(
        "--framework-case",
        required=True,
        help="existing case folder name under 03_Test_Cases",
    )
    export_parser.add_argument(
        "--evidence-request",
        type=Path,
        required=True,
        help="existing V0.3 Evidence Request inside the target case",
    )
    export_parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="separate root for deterministic governed export sets",
    )
    export_parser.add_argument(
        "--case-id",
        help=(
            "explicit pre-existing synthetic case_id for a standalone source run; "
            "cannot reassign an existing package case_id"
        ),
    )
    export_parser.add_argument(
        "--author-or-source",
        default="RAG Evidence Harness",
        help="attributable source recorded in governed object envelopes",
    )
    reuse_parser = subparsers.add_parser(
        "reuse-v03",
        help="derive new client-scoped V0.3 evidence from a synthetic export",
    )
    reuse_parser.add_argument(
        "--source-export",
        type=Path,
        required=True,
        help="path to an existing validated synthetic governed export set",
    )
    reuse_parser.add_argument(
        "--framework-root",
        type=Path,
        required=True,
        help="path to the extracted frozen Technical Prototype V0.3 root",
    )
    reuse_parser.add_argument(
        "--framework-case",
        required=True,
        help="existing client case folder name under 03_Test_Cases",
    )
    reuse_parser.add_argument(
        "--evidence-request",
        type=Path,
        required=True,
        help="existing client Evidence Request inside the target case",
    )
    reuse_parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="separate root for deterministic governed reuse export sets",
    )
    reuse_parser.add_argument(
        "--author-or-source",
        default="RAG Evidence Harness",
        help="attributable source recorded in governed object envelopes",
    )
    return parser


def _load_configuration(
    path: Path,
    *,
    case_id: str | None,
    standalone: bool,
) -> RunConfiguration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("configuration root must be a JSON object")
    if case_id is not None:
        payload["case_id"] = case_id
    elif standalone:
        payload["case_id"] = None
    return RunConfiguration.model_validate(payload)


def _build_services(configuration: RunConfiguration) -> tuple[Any, Any]:
    try:
        from rag_harness.services.embeddings import OllamaEmbeddingService
        from rag_harness.services.vector_store import QdrantVectorStore
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "RAG service dependencies are unavailable; install the 'rag' extra"
        ) from error

    return (
        OllamaEmbeddingService(
            model=configuration.embedding_model,
            dimension=configuration.embedding_dimension,
        ),
        QdrantVectorStore.from_configuration(configuration),
    )


def exit_code_for(package: RunPackage) -> int:
    if package.status == "completed" and package.experiment_result == "pass":
        return EXIT_PASS
    if package.status == "completed" and package.experiment_result == "fail":
        return EXIT_EXPERIMENT_FAIL
    if package.status == "blocked":
        return EXIT_BLOCKED
    return EXIT_EXECUTION_FAILED


def _write_json(payload: dict[str, Any], *, error: bool = False) -> None:
    print(
        json.dumps(payload, sort_keys=True),
        file=sys.stderr if error else sys.stdout,
    )


def _input_error(error: Exception) -> int:
    _write_json(
        {
            "exit_code": EXIT_INPUT_ERROR,
            "status": "input_error",
            "error_type": type(error).__name__,
            "message": str(error).replace("\n", " ").strip()[:1000],
        },
        error=True,
    )
    return EXIT_INPUT_ERROR


def _run(args: argparse.Namespace) -> int:
    try:
        configuration = _load_configuration(
            args.config,
            case_id=args.case_id,
            standalone=args.standalone,
        )
        definition = get_experiment(configuration.experiment_id)
        embeddings, vector_store = _build_services(configuration)
        workspace, package = execute_experiment(
            runs_root=args.runs_root,
            definition=definition,
            configuration=configuration,
            corpus=CONTROLLED_CORPUS,
            embeddings=embeddings,
            vector_store=vector_store,
            run_id=args.run_id,
        )
    except (
        FileExistsError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValidationError,
        ValueError,
    ) as error:
        return _input_error(error)

    exit_code = exit_code_for(package)
    _write_json(
        {
            "run_id": package.run_id,
            "case_id": package.case_id,
            "status": package.status,
            "experiment_result": package.experiment_result,
            "run_package_path": str(workspace.package_path.resolve()),
            "exit_code": exit_code,
            "governed_v0_3_export_claimed": package.governed_v0_3_export_claimed,
        }
    )
    return exit_code


def _verify(args: argparse.Namespace) -> int:
    verification = verify_run_package(args.package)
    exit_code = (
        EXIT_PASS
        if verification.verification_status == "verified"
        else EXIT_VERIFICATION_FAILED
    )
    payload = verification.model_dump(mode="json")
    payload["exit_code"] = exit_code
    _write_json(payload)
    return exit_code


def _export_v03(args: argparse.Namespace) -> int:
    try:
        result = export_verified_run_v03(
            package_path=args.package,
            framework_root=args.framework_root,
            framework_case_name=args.framework_case,
            evidence_request_path=args.evidence_request,
            output_root=args.output_root,
            author_or_source=args.author_or_source,
            target_case_id=args.case_id,
        )
    except (
        FileExistsError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValidationError,
        ValueError,
    ) as error:
        _write_json(
            {
                "exit_code": EXIT_GOVERNED_EXPORT_FAILED,
                "status": "governed_export_failed",
                "error_type": type(error).__name__,
                "message": str(error).replace("\n", " ").strip()[:1000],
                "source_evidence_changed": False,
            },
            error=True,
        )
        return EXIT_GOVERNED_EXPORT_FAILED
    payload = result.model_dump(mode="json")
    payload["exit_code"] = EXIT_PASS
    _write_json(payload)
    return EXIT_PASS


def _reuse_v03(args: argparse.Namespace) -> int:
    try:
        result = reuse_synthetic_export_v03(
            source_export_dir=args.source_export,
            framework_root=args.framework_root,
            framework_case_name=args.framework_case,
            evidence_request_path=args.evidence_request,
            output_root=args.output_root,
            author_or_source=args.author_or_source,
        )
    except (
        FileExistsError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValidationError,
        ValueError,
    ) as error:
        _write_json(
            {
                "exit_code": EXIT_GOVERNED_EXPORT_FAILED,
                "status": "governed_reuse_failed",
                "error_type": type(error).__name__,
                "message": str(error).replace("\n", " ").strip()[:1000],
                "source_evidence_changed": False,
            },
            error=True,
        )
        return EXIT_GOVERNED_EXPORT_FAILED
    payload = result.model_dump(mode="json")
    payload["exit_code"] = EXIT_PASS
    _write_json(payload)
    return EXIT_PASS


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "verify":
        return _verify(args)
    if args.command == "export-v03":
        return _export_v03(args)
    if args.command == "reuse-v03":
        return _reuse_v03(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

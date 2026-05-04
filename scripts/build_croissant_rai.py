"""Build the NeurIPS E&D Croissant+RAI metadata file.

Hugging Face provides the core Croissant JSON-LD at:

    https://huggingface.co/api/datasets/<repo-id>/croissant

NeurIPS E&D additionally requires minimal Responsible AI (RAI) fields in
the Croissant file uploaded to OpenReview. This script downloads (or
reads) the HF-generated core file, augments it with dataset-level RAI and
provenance fields, and writes a completed JSON-LD file for submission.

The output is a real submission artifact, not a placeholder.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import urllib.request
import urllib.error
from pathlib import Path


DEFAULT_REPO_ID = "anon-neuripsed26/multisource-memory-benchmark"
DEFAULT_ARCHIVE_PATH = "archives/multisource-memory-benchmark-data-v0.1.0.zip"
DEFAULT_ARCHIVE_SHA256 = (
    "7f1260b1ab6456f46935fa5de582be143a68e19ce6ae5266382b5ee85123a299"
)
DEFAULT_ARCHIVE_SIZE_BYTES = 37692022


def _load_json(path_or_url: str) -> dict:
    if path_or_url.startswith(("http://", "https://")):
        try:
            try:
                import certifi  # type: ignore

                context = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                context = None
            with urllib.request.urlopen(path_or_url, timeout=60, context=context) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise SystemExit(
                "Could not download the HF Croissant endpoint with Python's "
                f"HTTPS stack: {e}. Workaround: download it with curl and pass "
                "`--input /path/to/hf_croissant.json`."
            ) from e
    return json.loads(Path(path_or_url).read_text(encoding="utf-8"))


def _append_unique_by_id(items: list, additions: list[dict]) -> None:
    """Append JSON-LD nodes without duplicating existing @id entries."""

    seen = {item.get("@id") for item in items if isinstance(item, dict)}
    for item in additions:
        item_id = item.get("@id")
        if item_id not in seen:
            items.append(item)
            seen.add(item_id)


def _remove_invalid_sha256(metadata: dict) -> None:
    """Drop non-checksum sha256 values inherited from generated endpoints.

    Hugging Face's generated Croissant endpoint may use the ``sha256`` key on
    dynamic git-backed FileObjects with an explanatory issue URL instead of a
    checksum. That is helpful context for endpoint maintainers, but the
    OpenReview upload should only use ``sha256`` for actual 64-hex digests.
    """

    checksum = re.compile(r"^[0-9a-fA-F]{64}$")

    def visit(node: object) -> None:
        if isinstance(node, dict):
            value = node.get("sha256")
            if value is not None and (
                not isinstance(value, str) or checksum.fullmatch(value) is None
            ):
                del node["sha256"]
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(metadata)


def _move_ids_first(items: list, priority_ids: list[str]) -> None:
    """Move human-facing file-artifact entries before HF viewer entries."""

    rank = {item_id: i for i, item_id in enumerate(priority_ids)}

    def key(item: object) -> tuple[int, int]:
        if not isinstance(item, dict):
            return (len(rank), 0)
        item_id = item.get("@id") or item.get("name")
        return (rank.get(item_id, len(rank)), 0)

    items.sort(key=key)


def _remove_hf_viewer_schema(metadata: dict) -> None:
    """Remove HF auto-Parquet viewer nodes from the OpenReview metadata.

    The Hugging Face endpoint describes four one-row split-summary Parquet
    configs. Those rows are useful for the HF dataset viewer, but the NeurIPS
    artifact is the checksum-verified JSON file tree. Keeping the viewer nodes
    first makes the Croissant file look smaller than the released benchmark and
    also inherits a dynamic git FileObject without a real checksum.
    """

    viewer_distribution_ids = {
        "repo",
        "parquet-files-for-config-s20260321",
        "parquet-files-for-config-s20260322",
        "parquet-files-for-config-s20260323",
        "parquet-files-for-config-s20260324",
    }
    viewer_record_set_ids = {
        "s20260321_splits",
        "s20260321",
        "s20260322_splits",
        "s20260322",
        "s20260323_splits",
        "s20260323",
        "s20260324_splits",
        "s20260324",
    }

    metadata["distribution"] = [
        item
        for item in metadata.get("distribution", [])
        if not (isinstance(item, dict) and item.get("@id") in viewer_distribution_ids)
    ]
    metadata["recordSet"] = [
        item
        for item in metadata.get("recordSet", [])
        if not (isinstance(item, dict) and item.get("@id") in viewer_record_set_ids)
    ]


def _field(
    record_id: str,
    name: str,
    *,
    file_set_id: str,
    regex: str | None = None,
    data_type: str = "sc:Text",
) -> dict:
    """Create a Croissant field sourced from a file-set full path."""

    source: dict = {
        "fileSet": {"@id": file_set_id},
        "extract": {"fileProperty": "fullpath"},
    }
    if regex is not None:
        source["transform"] = {"regex": regex}
    return {
        "@type": "cr:Field",
        "@id": f"{record_id}/{name}",
        "name": name,
        "dataType": data_type,
        "source": source,
    }


def add_file_artifact_schema(metadata: dict, repo_id: str) -> dict:
    """Describe the actual file-tree artifact in addition to HF viewer rows.

    Hugging Face's automatically generated Croissant endpoint describes the
    converted Parquet viewer. Our benchmark is primarily a file artifact, so we
    add stable FileSet/RecordSet entries for the directories reviewers fetch:
    raw benchmark files, cached results, GPT/Gemini atoms, and frozen method
    outputs. These entries are intentionally path-level schemas rather than
    exhaustive JSON subfield enumerations, which keeps the Croissant file small
    while documenting the complete release surface.
    """

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    distribution = metadata.setdefault("distribution", [])
    record_sets = metadata.setdefault("recordSet", [])

    _append_unique_by_id(
        distribution,
        [
            {
                "@type": "cr:FileObject",
                "@id": "release-archive-zip",
                "name": "Checksum-verified release ZIP archive",
                "description": (
                    "Compressed reviewer-friendly mirror containing the heavy "
                    "data directories benchmark/, extracted_atoms/, and "
                    "method_outputs/. The fetch script verifies this SHA256 "
                    "checksum before extraction, then downloads the small "
                    "top-level metadata files separately."
                ),
                "contentUrl": f"{dataset_url}/resolve/main/{DEFAULT_ARCHIVE_PATH}",
                "encodingFormat": "application/zip",
                "contentSize": str(DEFAULT_ARCHIVE_SIZE_BYTES),
                "sha256": DEFAULT_ARCHIVE_SHA256,
            },
            {
                "@type": "cr:FileSet",
                "@id": "benchmark-seed-json-files",
                "name": "benchmark/seeds persona and source JSON files",
                "description": (
                    "Per-seed synthetic personas, latent event tables, source "
                    "streams, split assignments, and deterministic labels."
                ),
                "containedIn": {"@id": "release-archive-zip"},
                "encodingFormat": "application/json",
                "includes": "benchmark/seeds/**/*.json",
            },
            {
                "@type": "cr:FileSet",
                "@id": "benchmark-nl-render-md-files",
                "name": "benchmark/seeds natural-language memory renders",
                "description": (
                    "Templated natural-language memory documents used by "
                    "extraction, LLM-Direct, and Schema-Aware baselines."
                ),
                "containedIn": {"@id": "release-archive-zip"},
                "encodingFormat": "text/markdown",
                "includes": "benchmark/seeds/*/nl_renders/*.md",
            },
            {
                "@type": "cr:FileSet",
                "@id": "benchmark-result-json-files",
                "name": "benchmark/results JSON files",
                "description": (
                    "Aggregate result tables, confidence intervals, robustness "
                    "summaries, and paper-reproduction caches."
                ),
                "containedIn": {"@id": "release-archive-zip"},
                "encodingFormat": "application/json",
                "includes": "benchmark/results/**/*.json",
            },
            {
                "@type": "cr:FileSet",
                "@id": "extracted-atom-json-files",
                "name": "extracted_atoms GPT-5.4 test-split JSON files",
                "description": (
                    "Canonical frozen GPT-5.4 extracted atoms for held-out test "
                    "personas; train/calibration direct-readout atoms are "
                    "deterministically reconstructed by the code. These cached "
                    "outputs are redistributed for reproducibility subject to "
                    "upstream provider terms, not relicensed as CC-BY-4.0."
                ),
                "containedIn": {"@id": "release-archive-zip"},
                "encodingFormat": "application/json",
                "includes": "extracted_atoms/**/*.json",
            },
            {
                "@type": "cr:FileSet",
                "@id": "method-output-json-files",
                "name": "method_outputs frozen LLM and method JSON files",
                "description": (
                    "Frozen direct, schema-aware, few-shot, structured-input, "
                    "and cross-extractor method outputs used for API-free "
                    "reproduction. These cached outputs are redistributed for "
                    "reproducibility subject to upstream provider terms, not "
                    "relicensed as CC-BY-4.0."
                ),
                "containedIn": {"@id": "release-archive-zip"},
                "encodingFormat": "application/json",
                "includes": "method_outputs/**/*.json",
            },
        ],
    )

    _append_unique_by_id(
        record_sets,
        [
            {
                "@type": "cr:RecordSet",
                "@id": "benchmark_seed_file_records",
                "name": "benchmark_seed_file_records",
                "description": (
                    "One record per JSON file under benchmark/seeds. These files "
                    "define the synthetic personas, latent 30-day event tables, "
                    "source projections, split assignments, and ground-truth "
                    "labels."
                ),
                "key": {"@id": "benchmark_seed_file_records/relative_path"},
                "field": [
                    _field(
                        "benchmark_seed_file_records",
                        "relative_path",
                        file_set_id="benchmark-seed-json-files",
                    ),
                    _field(
                        "benchmark_seed_file_records",
                        "seed",
                        file_set_id="benchmark-seed-json-files",
                        regex=r"benchmark/seeds/(s[0-9]{8})/.*",
                    ),
                    _field(
                        "benchmark_seed_file_records",
                        "persona_id",
                        file_set_id="benchmark-seed-json-files",
                        regex=r"benchmark/seeds/s[0-9]{8}/([^/]+)/.*",
                    ),
                    _field(
                        "benchmark_seed_file_records",
                        "component",
                        file_set_id="benchmark-seed-json-files",
                        regex=r"benchmark/seeds/s[0-9]{8}/(?:[^/]+/)?(.+)$",
                    ),
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "benchmark_nl_render_file_records",
                "name": "benchmark_nl_render_file_records",
                "description": (
                    "One record per templated natural-language memory render "
                    "under benchmark/seeds/<seed>/nl_renders/."
                ),
                "key": {"@id": "benchmark_nl_render_file_records/relative_path"},
                "field": [
                    _field(
                        "benchmark_nl_render_file_records",
                        "relative_path",
                        file_set_id="benchmark-nl-render-md-files",
                    ),
                    _field(
                        "benchmark_nl_render_file_records",
                        "seed",
                        file_set_id="benchmark-nl-render-md-files",
                        regex=r"benchmark/seeds/(s[0-9]{8})/nl_renders/.*",
                    ),
                    _field(
                        "benchmark_nl_render_file_records",
                        "persona_id",
                        file_set_id="benchmark-nl-render-md-files",
                        regex=r"benchmark/seeds/s[0-9]{8}/nl_renders/([^/]+)\.md$",
                    ),
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "benchmark_result_file_records",
                "name": "benchmark_result_file_records",
                "description": (
                    "One record per cached aggregate JSON result under "
                    "benchmark/results."
                ),
                "key": {"@id": "benchmark_result_file_records/relative_path"},
                "field": [
                    _field(
                        "benchmark_result_file_records",
                        "relative_path",
                        file_set_id="benchmark-result-json-files",
                    ),
                    _field(
                        "benchmark_result_file_records",
                        "filename",
                        file_set_id="benchmark-result-json-files",
                        regex=r"benchmark/results/(.+)$",
                    ),
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "extracted_atom_file_records",
                "name": "extracted_atom_file_records",
                "description": (
                    "One record per canonical GPT-5.4 extracted-atom JSON file "
                    "for test-split personas."
                ),
                "key": {"@id": "extracted_atom_file_records/relative_path"},
                "field": [
                    _field(
                        "extracted_atom_file_records",
                        "relative_path",
                        file_set_id="extracted-atom-json-files",
                    ),
                    _field(
                        "extracted_atom_file_records",
                        "seed",
                        file_set_id="extracted-atom-json-files",
                        regex=r"extracted_atoms/(s[0-9]{8})/.*",
                    ),
                    _field(
                        "extracted_atom_file_records",
                        "persona_id",
                        file_set_id="extracted-atom-json-files",
                        regex=r"extracted_atoms/s[0-9]{8}/([^/]+)\.json$",
                    ),
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "method_output_file_records",
                "name": "method_output_file_records",
                "description": (
                    "One record per frozen method-output JSON file, including "
                    "LLM direct/schema/few-shot outputs, structured-input "
                    "diagnostics, and Gemini cross-extractor artifacts."
                ),
                "key": {"@id": "method_output_file_records/relative_path"},
                "field": [
                    _field(
                        "method_output_file_records",
                        "relative_path",
                        file_set_id="method-output-json-files",
                    ),
                    _field(
                        "method_output_file_records",
                        "provider_or_group",
                        file_set_id="method-output-json-files",
                        regex=r"method_outputs/([^/]+)/.*",
                    ),
                    _field(
                        "method_output_file_records",
                        "seed",
                        file_set_id="method-output-json-files",
                        regex=r"method_outputs/[^/]+/(s[0-9]{8})/.*",
                    ),
                ],
            },
        ],
    )

    return metadata


def add_rai_fields(metadata: dict, repo_id: str) -> dict:
    """Add minimal NeurIPS RAI fields to an HF-generated Croissant file."""

    ctx = metadata.setdefault("@context", {})
    if not isinstance(ctx, dict):
        raise TypeError("@context must be a JSON object for augmentation")
    ctx.setdefault("rai", "http://mlcommons.org/croissant/RAI/")
    ctx.setdefault("prov", "http://www.w3.org/ns/prov#")

    # Keep the HF-generated core conformance field intact and explicitly
    # declare RAI conformance using the prefix spelling in the RAI spec.
    metadata["dct:conformsTo"] = "http://mlcommons.org/croissant/RAI/1.0"

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    code_url = "https://github.com/anon-neuripsed26/multisource-memory-benchmark"

    metadata.setdefault("version", "0.1.0")
    metadata.setdefault("datePublished", "2026-05-02")
    metadata.setdefault(
        "citeAs",
        (
            "@misc{anonymous_2026_selective_qa_memory,\n"
            "  title={Selective QA over Conflicting Multi-Source Personal Memory: "
            "A Diagnostic Testbed and Method Comparison},\n"
            "  author={Anonymous Authors},\n"
            "  year={2026},\n"
            "  note={Anonymous submission, NeurIPS 2026 Evaluations & Datasets Track}\n"
            "}"
        ),
    )

    metadata["rai:dataLimitations"] = [
        (
            "The benchmark is synthetic and diagnostic. It does not contain "
            "field-collected personal-memory records and should not be used "
            "as evidence that a method will generalize to real users without "
            "external validation."
        ),
        (
            "The released question set is intentionally narrow: 18 closed-class "
            "templates over five topics (work, diet, social, sleep, exercise) "
            "and English, US-style natural-language renders. Broader life "
            "domains, multilingual settings, and free-form disclosure are out "
            "of scope."
        ),
        (
            "The data-generation process uses simplified persona distributions "
            "and controlled source distortions. These distortions are stress-test "
            "assumptions, not calibrated empirical effect sizes for real "
            "populations."
        ),
        (
            "Cached LLM outputs are included only for exact reproduction of the "
            "paper tables. Downstream use of those outputs should respect the "
            "upstream model-provider terms linked in the dataset card."
        ),
    ]
    metadata["rai:dataBiases"] = [
        (
            "Source distortions are deliberately injected: planner streams are "
            "optimistic, self-reports are topic-dependent, device logs contain "
            "dropout, and profile memories can be stale or idealized. These "
            "biases are benchmark mechanisms rather than incidental annotation "
            "artifacts."
        ),
        (
            "Persona attributes, activity distributions, names, occupations, and "
            "measurement conventions are synthetic and simplified. The templates "
            "are Western-leaning and English-only, so results should not be read "
            "as demographic or cultural coverage claims."
        ),
    ]
    metadata["rai:personalSensitiveInformation"] = [
        (
            "No real personal or sensitive information is included. All names, "
            "personas, daily events, source streams, and natural-language memory "
            "renders are synthetic."
        ),
        (
            "Synthetic personas include simulated age, occupation, diet, exercise, "
            "sleep, work, and social-behavior attributes solely to support the "
            "benchmark's controlled question templates."
        ),
    ]
    metadata["rai:dataUseCases"] = [
        (
            "Intended use: diagnostic evaluation of selective question-answering "
            "and conflict-resolution methods over controlled multi-source "
            "personal-memory evidence."
        ),
        (
            "Intended use: studying answer/SKIP trade-offs, source-bias fusion, "
            "and resolver-versus-input decompositions under known synthetic "
            "distortions."
        ),
        (
            "Not intended for training production personal-memory assistants, "
            "evaluating real-user privacy risks, or making demographic claims "
            "about real populations."
        ),
    ]
    metadata["rai:dataSocialImpact"] = (
        "Potential positive impact: the benchmark encourages personal-memory "
        "agents to abstain when evidence is insufficient rather than hallucinate "
        "answers, and helps diagnose when systems over-trust biased sources. "
        "Potential negative impact: techniques developed on the benchmark could "
        "be misapplied to infer sensitive habits or preferences from real users. "
        "Mitigations: the released data are fully synthetic, the paper frames "
        "deployment as requiring consent and external validation, and the dataset "
        "is documented as a diagnostic evaluation resource rather than a "
        "production-training corpus."
    )
    metadata["rai:hasSyntheticData"] = True

    metadata["rai:dataCollection"] = (
        "The dataset is generated by a deterministic Python data-generation "
        "pipeline. Each seed creates personas, a 30-day latent event table, five "
        "projected evidence streams, deterministic ground-truth answers, and "
        "templated natural-language renders."
    )
    metadata["rai:dataCollectionType"] = [
        "Synthetic data generation",
        "Software collection",
        "Deterministic simulation",
    ]
    metadata["rai:dataCollectionRawData"] = (
        "The raw data are the generated latent event tables and persona "
        "parameters under benchmark/seeds/<seed>/<persona_id>/event_table.json; "
        "they are not collected from real people."
    )
    metadata["rai:dataCollectionMissingData"] = (
        "Missingness is part of the synthetic design: device logs include field "
        "dropout and day-level missingness, while some sources lack fields for "
        "specific topics by construction. Ground-truth answers are complete."
    )
    metadata["rai:dataPreprocessingProtocol"] = [
        (
            "Structured source files are direct outputs of the projection "
            "pipeline. Natural-language renders are generated by a templated, "
            "non-LLM renderer from the structured sources."
        ),
        (
            "Frozen LLM-extracted atoms and cached method outputs are included "
            "for reproducibility; generation of new LLM outputs requires explicit "
            "API credentials and is not needed to reproduce the reported tables."
        ),
        (
            "For reviewer-friendly download, the heavy JSON file tree is also "
            "released as a checksum-verified ZIP archive under archives/. The "
            "archive contains the same benchmark/, extracted_atoms/, and "
            "method_outputs/ directories as the expanded Hugging Face tree."
        ),
    ]
    metadata["rai:dataAnnotationProtocol"] = (
        "Labels are deterministic ground-truth answers computed by question-"
        "specific aggregation functions from the latent event table and, for "
        "nine templates, structured source annotations. There are no human "
        "annotators."
    )
    metadata["rai:dataAnnotationAnalysis"] = [
        (
            "Ground-truth correctness is checked by deterministic re-execution "
            "and an internal independent label-rule reimplementation with 100% "
            "label agreement, as documented in the paper appendix and DATASHEET."
        )
    ]
    metadata["rai:annotationsPerItem"] = (
        "One deterministic label per (persona, seed, question) instance; no "
        "human labels or inter-annotator agreement are involved."
    )
    metadata["rai:machineAnnotationTools"] = [
        "survey2agent deterministic data-generation and ground-truth pipeline",
        "templated natural-language renderer",
        "frozen LLM extractor outputs for held-out test-split atoms",
    ]
    metadata["rai:dataReleaseMaintenancePlan"] = (
        "The anonymous Hugging Face artifact is maintained for review. Upon "
        "acceptance, it will be replaced by a permanent de-anonymized release "
        "with archival DOI; existing seeds/questions will not be silently "
        "changed."
    )

    metadata["prov:wasDerivedFrom"] = [
        {
            "@type": "prov:Entity",
            "name": "Synthetic seed configuration",
            "url": f"{dataset_url}/tree/main/benchmark/seeds",
            "description": (
                "Four deterministic seed directories s20260321--s20260324; "
                "no external source dataset or real-user record was used."
            ),
        }
    ]
    metadata["prov:wasGeneratedBy"] = [
        {
            "@type": "prov:Activity",
            "name": "Deterministic synthetic benchmark generation",
            "description": (
                "Persona sampling, latent 30-day event simulation, source "
                "projection, natural-language rendering, and deterministic "
                "ground-truth labeling."
            ),
            "url": code_url,
        },
        {
            "@type": "prov:Activity",
            "name": "Frozen model-output caching",
            "description": (
                "Cached LLM extraction and method-output artifacts were generated "
                "for byte-stable reproduction and uploaded with the dataset."
            ),
            "url": f"{dataset_url}/tree/main/method_outputs",
        },
    ]

    metadata = add_file_artifact_schema(metadata, repo_id)
    _remove_hf_viewer_schema(metadata)
    _move_ids_first(
        metadata.get("distribution", []),
        [
            "release-archive-zip",
            "benchmark-seed-json-files",
            "benchmark-nl-render-md-files",
            "benchmark-result-json-files",
            "extracted-atom-json-files",
            "method-output-json-files",
        ],
    )
    _move_ids_first(
        metadata.get("recordSet", []),
        [
            "benchmark_seed_file_records",
            "benchmark_nl_render_file_records",
            "benchmark_result_file_records",
            "extracted_atom_file_records",
            "method_output_file_records",
        ],
    )
    _remove_invalid_sha256(metadata)
    return metadata


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument(
        "--input",
        default=None,
        help="Input Croissant JSON or URL. Defaults to the HF /croissant endpoint.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/CROISSANT_RAI.json"),
        help="Output path for completed Croissant+RAI JSON.",
    )
    args = p.parse_args()

    source = args.input or f"https://huggingface.co/api/datasets/{args.repo_id}/croissant"
    metadata = add_rai_fields(_load_json(source), args.repo_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

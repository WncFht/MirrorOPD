"""Normalize explicit dataset profiles to the canonical backend parquet contract."""

import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from mirroropd.identity import canonical_json, content_id
from mirroropd.runtime.artifacts import atomic_json, file_digest


def column_value(row, path, *, required=True):
    value = row
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            if required:
                raise ValueError(f"Dataset row has no configured field: {path}")
            return None
        value = value[component]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if required:
            raise ValueError(f"Dataset field is null: {path}")
        return None
    return value


def messages(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value:
        raise ValueError("Configured prompt must be text or a nonempty message list")
    result = []
    for message in value:
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
        ):
            raise TypeError("Each prompt message requires string role/content")
        result.append(dict(message))
    return result


def read_frame(path):
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Dataset files must be parquet, jsonl or csv")


def prepare_data(dataset, directory):
    import pandas as pd

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 2, "splits": {}, "files": []}
    for split, inputs in (("train", dataset.train), ("validation", dataset.validation)):
        prepared = []
        sources = Counter()
        labels = 0
        for index, name in enumerate(inputs):
            path = Path(name).expanduser().resolve()
            digest = file_digest(path)
            records = []
            for row_index, row in enumerate(read_frame(path).to_dict(orient="records")):
                prompt = messages(column_value(row, dataset.columns.prompt))
                answer = column_value(
                    row,
                    dataset.columns.answer,
                    required=(split == "validation" and dataset.requires_validation_reference),
                )
                extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
                available = answer is not None and bool(extra.get("has_reference", True))
                if (
                    split == "validation"
                    and dataset.requires_validation_reference
                    and not available
                ):
                    raise ValueError("Validation row has no reference answer")
                if answer is None:
                    answer = ""
                elif not isinstance(answer, str):
                    answer = canonical_json(answer)
                source = dataset.source_name or column_value(row, dataset.columns.source)
                if not isinstance(source, str) or not source:
                    raise ValueError("Dataset data_source must be a nonempty string")
                identity = content_id({"artifact_sha256": digest, "row": row_index, "split": split})
                sources[source] += 1
                labels += int(available)
                records.append(
                    {
                        "data_source": source,
                        "prompt": prompt,
                        "ability": dataset.ability,
                        "reward_model": {"style": "rule", "ground_truth": answer},
                        "uid": identity,
                        "extra_info": extra
                        | {"split": split, "has_reference": available, "row_uid": identity},
                    }
                )
            if not records:
                raise ValueError(f"Dataset file is empty: {path}")
            target = directory / f"{split}-{index:03d}-{digest[:12]}.parquet"
            pd.DataFrame(records).to_parquet(target, index=False)
            prepared.append(str(target.resolve()))
            manifest["files"].append(
                {
                    "split": split,
                    "source_path": str(path),
                    "source_sha256": digest,
                    "prepared_path": str(target.resolve()),
                    "prepared_sha256": file_digest(target),
                    "rows": len(records),
                }
            )
        manifest["splits"][split] = {
            "paths": prepared,
            "rows": sum(sources.values()),
            "labeled_rows": labels,
            "sources": dict(sources),
        }
    manifest["dataset_id"] = content_id(
        {
            "schema_version": 2,
            "files": [
                {"sha256": item["source_sha256"], "split": item["split"]}
                for item in manifest["files"]
            ],
            "columns": asdict(dataset.columns),
            "source_name": dataset.source_name,
            "ability": dataset.ability,
            "requires_validation_reference": dataset.requires_validation_reference,
        }
    )
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def evaluation_questions(manifest, limit):
    total = manifest["splits"]["validation"]["rows"]
    return min(total, limit) if limit is not None else total

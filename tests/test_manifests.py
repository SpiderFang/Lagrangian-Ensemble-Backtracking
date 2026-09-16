"""情境 component 與巢狀邊界 manifest 的嚴格契約測試。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import yaml
from shapely.geometry import LineString, box, mapping

from lagrangian_backtracking.arrival_times import select_arrival_times
from lagrangian_backtracking.bed_residence import (
    BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
    BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
    BED_RESIDENCE_POLICY_ID,
    BED_RESIDENCE_SAMPLING_POLICY_ID,
    apply_bed_residence_sampling,
    sample_bed_residence_age_hours,
)
from lagrangian_backtracking.config import ProjectConfig, resolve_flow_domain_id
from lagrangian_backtracking.input_horizon import BED_RESIDENCE_INPUT_SCHEMA_VERSION
from lagrangian_backtracking.manifests import (
    load_arrival_time_manifest,
    load_boundary_geometries,
    load_material_manifest,
    load_receptor_arrival_initial_condition_manifest,
    load_receptor_manifest,
    load_scenario_inputs,
    resolve_manifest_path,
)
from lagrangian_backtracking.scenarios import (
    BASELINE_BEHAVIORS,
    ArrivalTime,
    Receptor,
    stable_identifier,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "lagrangian_backtracking.example.yaml"
SITES = {
    "gongliao": ("A", "northeast_taiwan_common_cache_v3"),
    "guishan": ("A", "northeast_taiwan_common_cache_v3"),
    "hsinchu": ("B", "hsinchu_cache_v3"),
    "houwan": ("C", "houwan_nmmba_cache_v3"),
    "lienchiang": ("D", "lienchiang_common_cache_v3"),
}


def _config(tmp_path: Path, *, with_paths: bool = False) -> ProjectConfig:
    """建立明確未啟用新沉底政策的 legacy fixture，供既有 manifest 測試使用。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["scenarios"].pop("bed_residence_time", None)
    payload["inputs"].pop("backtrack_support_days", None)
    if with_paths:
        payload["scenarios"]["material_manifest"] = "material.json"
        payload["scenarios"]["receptor_manifest"] = "receptor.json"
        payload["scenarios"]["arrival_time_manifest"] = "arrival.json"
        payload["scenarios"]["receptor_arrival_initial_condition_manifest"] = (
            "initial_conditions.json"
        )
        payload["geometry"]["domain_manifest"] = "domain.json"
        payload["geometry"]["local_domain_manifest"] = "local.json"
        payload["geometry"]["open_boundary_manifest"] = "open.json"
    else:
        payload["scenarios"]["receptor_arrival_initial_condition_manifest"] = None
    return ProjectConfig.model_validate(payload)


def _bed_config() -> ProjectConfig:
    """建立五站正式沉底 loader 測試設定，分離 180 日選時與 90 日運算支援。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["inputs"]["backtrack_support_days"] = 180
    payload["scenarios"]["bed_residence_time"].update(
        {
            "backtrack_mode": BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
            "supported_backtrack_modes": [
                BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
                BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
            ],
            "runtime_horizon_support_days": 90,
        }
    )
    return ProjectConfig.model_validate(payload)


def _write_json(path: Path, payload: object) -> None:
    """用一般 JSON writer 產生測試文件；NaN 測試會刻意保留非標準 token。"""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _provenance() -> dict[str, object]:
    """提供符合最低 provenance 契約的固定測試來源。"""

    return {
        "method_id": "synthetic_manifest_fixture_v1",
        "created_at_utc": "2026-08-28T00:00:00Z",
        "source_hashes": {"synthetic_input": "a" * 64},
    }


def _material_payload() -> dict[str, object]:
    """以既有 CLI 輸出 shape 建立 schema 2.0.0 material fixture。"""

    return {
        "schema_version": "2.0.0",
        "design_version": "design_baseline_v3_non_rising_a_v3_local20_20260909",
        "classification_source": "synthetic iOcean category fixture",
        "velocity_unit": "m s-1; z positive-up; all values must be strictly negative",
        "velocity_source": "design_sensitivity_grid_not_oca_measurement",
        "positive_or_zero_velocity_policy": "reject_config",
        "calibration_scope": "provisional_material_shape_proxy_pending_local_measurement",
        "records": [asdict(item) for item in BASELINE_BEHAVIORS],
    }


def _receptor_payload(*, per_site: int = 20, wrong_region: bool = False) -> dict[str, object]:
    """建立五站 receptor records；座標只供 manifest 型別測試，不取代 geometry fixture。"""

    records: list[dict[str, object]] = []
    for site_id, (region, _) in SITES.items():
        for index in range(per_site):
            records.append(
                {
                    "receptor_id": f"{site_id}_r{index:02d}",
                    "study_site_id": site_id,
                    "analysis_region_id": "B" if wrong_region and index == 0 else region,
                    "lon": 121.0 + index * 0.001,
                    "lat": 24.0 + index * 0.001,
                    "z_m_positive_up": -5.0 - index,
                    "vertical_id": f"z{index % 4}",
                    "metadata": {"candidate_rank": index},
                }
            )
    return {
        "manifest_kind": "receptor_manifest",
        "schema_version": "1.0.0",
        "status": "approved",
        "design_version": "design_baseline_v3_non_rising_a_v3_local20_20260909",
        "coordinate_reference": "EPSG:4326",
        "vertical_reference": "z_m_positive_up",
        "generation_method_id": "synthetic_receptor_fixture_v1",
        "provenance": _provenance(),
        "records": records,
    }


def _arrival_payload(*, per_site: int = 50, wrong_year: bool = False) -> dict[str, object]:
    """依正式 selector 的 48+2 契約建立逐站 records，pilot 可取其前段子集。

    前 48 筆完整交叉兩年、四季、兩種潮差代理與三種潮內相位；最後兩筆分別為高波及
    強流事件。時間使用各季代表月份且站內唯一，相同世界協調時間（UTC）可跨站重用。
    """

    records: list[dict[str, object]] = []
    season_month = {"DJF": 1, "MAM": 4, "JJA": 7, "SON": 10}
    for site_id in SITES:
        site_records: list[dict[str, object]] = []
        index = 0
        for year in (2024, 2025):
            for season, month in season_month.items():
                for tide_index, tide_class in enumerate(("spring_proxy", "neap_proxy")):
                    for phase_index, phase in enumerate(
                        ("fastest_rising", "fastest_falling", "slack_proxy")
                    ):
                        day = 1 + tide_index * 3 + phase_index
                        time_ns = int(
                            datetime(year, month, day, tzinfo=UTC).timestamp() * 1_000_000_000
                        )
                        site_records.append(
                            {
                                "arrival_time_id": f"{site_id}_t{index:02d}",
                                "study_site_id": site_id,
                                "time_utc_ns": time_ns,
                                "year": year,
                                "season": season,
                                "tide_class": tide_class,
                                "phase_or_event": phase,
                                "metadata": {"selection_rank": index},
                            }
                        )
                        index += 1
        for event_index, (year, month, season, event) in enumerate(
            (
                (2024, 2, "DJF", "high_wave_event"),
                (2025, 5, "MAM", "strong_current_event"),
            )
        ):
            time_ns = int(
                datetime(year, month, 20 + event_index, tzinfo=UTC).timestamp() * 1_000_000_000
            )
            site_records.append(
                {
                    "arrival_time_id": f"{site_id}_t{index:02d}",
                    "study_site_id": site_id,
                    "time_utc_ns": time_ns,
                    "year": year,
                    "season": season,
                    "tide_class": "event",
                    "phase_or_event": event,
                    "metadata": {"selection_rank": index},
                }
            )
            index += 1
        selected = site_records[:per_site]
        if wrong_year and selected:
            selected[0]["year"] = 2025
        records.extend(selected)
    return {
        "manifest_kind": "arrival_time_manifest",
        "schema_version": "1.0.0",
        "status": "approved",
        "design_version": "design_baseline_v3_non_rising_a_v3_local20_20260909",
        "time_standard": "UTC",
        "selection_method_id": "synthetic_arrival_fixture_v1",
        "provenance": _provenance(),
        "records": records,
    }


def _bed_arrival_payload(config: ProjectConfig) -> dict[str, object]:
    """以 48+2 observation anchor fixture 建立可由 formal bed loader 重算的母體。"""

    design_version = config.design_version
    observations_by_site: dict[str, list[ArrivalTime]] = {site_id: [] for site_id in SITES}
    for row in _arrival_payload()["records"]:
        assert isinstance(row, dict)
        site_id = str(row["study_site_id"])
        observation_ns = int(row["time_utc_ns"])
        tide_class = str(row["tide_class"])
        phase = str(row["phase_or_event"])
        # 龜山島是貢寮 48+2 UTC 的 paired clone，event identity 也沿用五欄來源政策；
        # 一般站點 event identity 則由四欄組成，與 production selector 一致。
        paired_a_clone = site_id == "guishan"
        identity_fields = (
            [site_id, str(observation_ns), tide_class, phase, design_version]
            if paired_a_clone or tide_class != "event"
            else [site_id, str(observation_ns), phase, design_version]
        )
        metadata: dict[str, float | int | str] = {"selection_rank": int(row["metadata"]["selection_rank"])}
        if paired_a_clone:
            metadata.update(
                {
                    "shared_A_forcing_utc": "true",
                    "shared_A_forcing_reference_site": "gongliao",
                    "shared_A_forcing_policy": "gongliao_paired_utc_reference_v1",
                }
            )
        observations_by_site[site_id].append(
            ArrivalTime(
                arrival_time_id=stable_identifier("arr", identity_fields),
                study_site_id=site_id,
                time_utc_ns=observation_ns,
                year=int(row["year"]),
                season=str(row["season"]),
                tide_class=tide_class,
                phase_or_event=phase,
                metadata=metadata,
            )
        )

    bed = config.scenarios.bed_residence_time
    assert bed is not None
    offsets = sample_bed_residence_age_hours(
        maximum_age_days=bed.maximum_age_days,
        sample_count=bed.sample_count_per_site,
        seed=bed.sampling_seed,
    )
    deposition_records = [
        asdict(arrival)
        for site_id in sorted(observations_by_site)
        for arrival in apply_bed_residence_sampling(
            observations_by_site[site_id],
            age_hours=offsets,
            sampling_seed=bed.sampling_seed,
            maximum_age_days=bed.maximum_age_days,
            design_version=design_version,
        )
    ]
    age_vector_hash = sha256(
        json.dumps(list(offsets), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_kind": "arrival_time_manifest",
        "schema_version": BED_RESIDENCE_INPUT_SCHEMA_VERSION,
        "status": "approved",
        "design_version": design_version,
        "time_standard": "UTC",
        "selection_method_id": (
            "server_v3_48_strata_plus_two_observation_anchors_then_random_deposition_v1"
        ),
        "provenance": {
            "method_id": (
                "server_v3_48_strata_plus_two_observation_anchors_then_random_deposition_v1"
            ),
            "created_at_utc": "2026-09-16T00:00:00Z",
            "source_hashes": {"synthetic_input": "a" * 64},
            "bed_residence_sampling": {
                "policy_id": BED_RESIDENCE_POLICY_ID,
                "sampling_policy_id": BED_RESIDENCE_SAMPLING_POLICY_ID,
                "maximum_age_days": bed.maximum_age_days,
                "sample_count_per_site": bed.sample_count_per_site,
                "sampling_seed": bed.sampling_seed,
                "shared_age_offsets_across_sites": True,
                "age_offsets_hours_sha256": age_vector_hash,
                "selection_support_days": 180,
                "runtime_support_days": 90,
                "pre_window_policy": bed.pre_window_policy,
                "observation_anchor_selection_method_id": (
                    "server_v3_48_strata_plus_two_events_gap_safe_nww_metric_location_v2"
                ),
            },
        },
        "records": deposition_records,
    }


def _initial_condition_payload(
    config: ProjectConfig,
    *,
    receptors: dict[str, object] | None = None,
    arrivals: dict[str, object] | None = None,
    status: str = "approved",
    formal: bool = True,
) -> dict[str, object]:
    """建立 5,000 筆 dynamic pair fixture，不讀 OCM，只模擬已產出的 OCM 摘要。

    每一筆以 eta、bed 與四個 vertical_id 的固定比例產生公尺制 z；arrival index 會改變
    eta，因此同一 receptor 在不同 UTC 具不同 actual z。此 fixture 特意把十種 material
    排除在外，確認 pair manifest 的資料量是 100×50 而非 50,000。
    """

    receptor_value = receptors or _receptor_payload()
    arrival_value = arrivals or _arrival_payload()
    receptor_rows = receptor_value["records"]
    arrival_rows = arrival_value["records"]
    assert isinstance(receptor_rows, list)
    assert isinstance(arrival_rows, list)
    arrivals_by_site: dict[str, list[dict[str, object]]] = {site: [] for site in SITES}
    for row in arrival_rows:
        assert isinstance(row, dict)
        arrivals_by_site[str(row["study_site_id"])].append(row)
    fractions = {"z0": 0.10, "z1": 0.40, "z2": 0.70, "z3": 0.90}
    records: list[dict[str, object]] = []
    for receptor_index, receptor_value_row in enumerate(receptor_rows):
        assert isinstance(receptor_value_row, dict)
        site_id = str(receptor_value_row["study_site_id"])
        region = str(receptor_value_row["analysis_region_id"])
        vertical_id = str(receptor_value_row["vertical_id"])
        fraction = fractions[vertical_id]
        flow_id = resolve_flow_domain_id(config, region, formal=formal)
        for arrival_index, arrival_value_row in enumerate(arrivals_by_site[site_id]):
            time_ns = int(arrival_value_row["time_utc_ns"])
            utc = datetime.fromtimestamp(time_ns / 1_000_000_000, tz=UTC)
            eta = 1.0 + (arrival_index % 5) * 0.01
            bed = -20.0 - (receptor_index % 2) * 0.1
            water = eta - bed
            z = bed + water * fraction
            records.append(
                {
                    "receptor_id": receptor_value_row["receptor_id"],
                    "arrival_time_id": arrival_value_row["arrival_time_id"],
                    "study_site_id": site_id,
                    "analysis_region_id": region,
                    "flow_domain_id": flow_id,
                    "time_utc_ns": time_ns,
                    "vertical_id": vertical_id,
                    "z_m_positive_up": z,
                    "eta_m_positive_up": eta,
                    "bed_z_m_positive_up": bed,
                    "water_column_height_m": water,
                    "height_above_bed_m": z - bed,
                    "zcor_lower_m_positive_up": bed,
                    "zcor_upper_m_positive_up": eta,
                    "vertical_bracket_alpha": fraction,
                    "source_face_local_index": receptor_index,
                    "source_face_global_index": receptor_index + 1000,
                    "wetdry_elem_value": 0,
                    "wetdry_semantics_id": "schism_wetdry_elem_0_wet_1_dry",
                    "ocm_month_yyyymm": f"{utc.year:04d}{utc.month:02d}",
                    "ocm_source_time_index": arrival_index,
                    "ocm_time_origin": "observed",
                }
            )
    return {
        "manifest_kind": "receptor_arrival_initial_condition_manifest",
        "schema_version": "1.0.0",
        "status": status,
        "design_version": config.design_version,
        "vertical_reference": "z_m_positive_up",
        "time_standard": "UTC",
        "generation_method_id": "synthetic_ocm_pair_initial_condition_v1",
        "provenance": _provenance(),
        "records": records,
    }


def _pilot_dynamic_fixture(
    tmp_path: Path,
) -> tuple[ProjectConfig, tuple, tuple, Path, dict[str, object]]:
    """建立五站各一個 receptor×arrival 的 pilot dynamic fixture。"""

    config = _config(tmp_path)
    receptor_payload = _receptor_payload(per_site=1)
    arrival_payload = _arrival_payload(per_site=1)
    receptor_path = tmp_path / "pilot-receptors.json"
    arrival_path = tmp_path / "pilot-arrivals.json"
    initial_path = tmp_path / "pilot-initial-conditions.json"
    _write_json(receptor_path, receptor_payload)
    _write_json(arrival_path, arrival_payload)
    receptors = load_receptor_manifest(receptor_path, config, formal=False)
    arrivals = load_arrival_time_manifest(arrival_path, config, formal=False)
    initial_payload = _initial_condition_payload(
        config,
        receptors=receptor_payload,
        arrivals=arrival_payload,
        status="pilot",
        formal=False,
    )
    _write_json(initial_path, initial_payload)
    return config, receptors, arrivals, initial_path, initial_payload


def _geometry_payloads(
    *, a_flow_id: str | None = None
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """建立四域五站的 WGS84 polygon/line fixture，模擬 A 區兩站巢狀 local domain。

    ``a_flow_id`` 只替換 fixture 的 A 區識別碼，讓同一組幾何形狀能測試 pilot base ID
    與 formal expanded ID 的 resolver 綁定；它不改變座標、polygon topology 或站點數量。
    """

    domain_bounds = {
        "A": (121.3, 122.8, 24.6, 25.5),
        "B": (119.7, 121.2, 24.3, 25.2),
        "C": (120.1, 121.7, 21.5, 22.5),
        "D": (119.1, 120.8, 25.7, 26.7),
    }
    flow_by_region = {region: flow for _, (region, flow) in SITES.items() if region != "A"}
    flow_by_region["A"] = a_flow_id or SITES["gongliao"][1]
    domain_records: list[dict[str, object]] = []
    local_records: list[dict[str, object]] = []
    open_records: list[dict[str, object]] = []
    domain_shapes: dict[str, object] = {}
    for region, bounds in domain_bounds.items():
        polygon = box(bounds[0], bounds[2], bounds[1], bounds[3])
        geometry = mapping(polygon)
        flow_id = flow_by_region[region]
        domain_shapes[flow_id] = geometry
        domain_records.append(
            {
                "analysis_region_id": region,
                "flow_domain_id": flow_id,
                "geometry": geometry,
                "source_geometry_id": f"synthetic_flow_{region}",
            }
        )
        min_lon, max_lon, min_lat, max_lat = bounds
        open_records.append(
            {
                "owner_kind": "flow_domain",
                "owner_id": flow_id,
                "analysis_region_id": region,
                "segment_id": f"{region}_flow_south",
                "geometry": mapping(LineString([(min_lon, min_lat), (max_lon, min_lat)])),
                "source_geometry_id": f"synthetic_flow_open_{region}",
            }
        )
    local_bounds = {
        "gongliao": (121.7, 122.15, 24.9, 25.3),
        "guishan": (121.7, 122.15, 24.7, 24.95),
        "hsinchu": domain_bounds["B"],
        "houwan": domain_bounds["C"],
        "lienchiang": domain_bounds["D"],
    }
    for site_id, (region, _) in SITES.items():
        flow_id = flow_by_region[region]
        bounds = local_bounds[site_id]
        polygon = box(bounds[0], bounds[2], bounds[1], bounds[3])
        local_records.append(
            {
                "study_site_id": site_id,
                "analysis_region_id": region,
                "flow_domain_id": flow_id,
                "local_equals_flow": site_id not in {"gongliao", "guishan"},
                "geometry": mapping(polygon),
                "source_geometry_id": f"synthetic_local_{site_id}",
            }
        )
        if site_id in {"gongliao", "guishan"}:
            min_lon, max_lon, min_lat, _ = local_bounds[site_id]
            open_records.append(
                {
                    "owner_kind": "local_domain",
                    "owner_id": site_id,
                    "analysis_region_id": region,
                    "segment_id": f"{site_id}_local_south",
                    "geometry": mapping(LineString([(min_lon, min_lat), (max_lon, min_lat)])),
                    "source_geometry_id": f"synthetic_local_open_{site_id}",
                }
            )
    root_base = {
        "schema_version": "1.0.0",
        "status": "approved",
        "design_version": "design_baseline_v3_non_rising_a_v3_local20_20260909",
        "coordinate_reference": "EPSG:4326",
        "provenance": _provenance(),
    }
    return (
        {
            "manifest_kind": "domain_geometry_manifest",
            **root_base,
            "records": domain_records,
        },
        {
            "manifest_kind": "local_geometry_manifest",
            **root_base,
            "records": local_records,
        },
        {
            "manifest_kind": "open_boundary_manifest",
            **root_base,
            "records": open_records,
        },
    )


def test_material_cli_v2_roundtrip_and_strict_numeric_rules(tmp_path: Path) -> None:
    """既有 CLI schema 可讀回；NaN 與 bool numeric 不得穿過 loader。"""

    config = _config(tmp_path)
    path = tmp_path / "material.json"
    payload = _material_payload()
    _write_json(path, payload)
    records = load_material_manifest(path, config, formal=True)
    assert records == BASELINE_BEHAVIORS

    invalid = deepcopy(payload)
    invalid["records"][0]["settling_velocity_mps"] = True
    _write_json(path, invalid)
    with pytest.raises(ValueError, match="有限數值"):
        load_material_manifest(path, config)
    invalid["records"][0]["settling_velocity_mps"] = float("nan")
    _write_json(path, invalid)
    with pytest.raises(ValueError, match="非有限常數"):
        load_material_manifest(path, config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"unexpected": 1}), "未知"),
        (lambda payload: payload.update({"schema_version": "9.0.0"}), "schema_version"),
    ],
)
def test_material_unknown_key_and_schema_fail_fast(
    tmp_path: Path, mutation, message: str
) -> None:
    """material root 的未知欄位或未登錄版本不可被寬鬆讀取。"""

    payload = _material_payload()
    mutation(payload)
    path = tmp_path / "material.json"
    _write_json(path, payload)
    with pytest.raises(ValueError, match=message):
        load_material_manifest(path, _config(tmp_path))


def test_receptor_and_arrival_formal_pilot_coverage_and_cross_reference(tmp_path: Path) -> None:
    """pilot 可用站點子集，formal 則固定要求 20/50 與五站完整 coverage。"""

    config = _config(tmp_path)
    receptor_path = tmp_path / "receptor.json"
    arrival_path = tmp_path / "arrival.json"
    receptor = _receptor_payload(per_site=1)
    arrival = _arrival_payload(per_site=1)
    _write_json(receptor_path, receptor)
    _write_json(arrival_path, arrival)
    assert len(load_receptor_manifest(receptor_path, config, formal=False)) == 5
    assert len(load_arrival_time_manifest(arrival_path, config, formal=False)) == 5
    with pytest.raises(ValueError, match="formal"):
        load_receptor_manifest(receptor_path, config, formal=True)
    with pytest.raises(ValueError, match="formal"):
        load_arrival_time_manifest(arrival_path, config, formal=True)

    wrong = _receptor_payload(per_site=1, wrong_region=True)
    _write_json(receptor_path, wrong)
    with pytest.raises(ValueError, match="analysis_region_id"):
        load_receptor_manifest(receptor_path, config)
    wrong_arrival = _arrival_payload(per_site=1, wrong_year=True)
    _write_json(arrival_path, wrong_arrival)
    with pytest.raises(ValueError, match="UTC year"):
        load_arrival_time_manifest(arrival_path, config)


def test_select_arrival_times_output_passes_manifest_loader(tmp_path: Path) -> None:
    """正式 selector 的 DJF/MAM/JJA/SON 產物可直接通過同一資料類別的 loader。

    此回歸測試使用兩年六小時合成序列實際呼叫 ``select_arrival_times``，避免 fixture 自己
    發明 season 或 phase label 後，與正式 selector 漂移卻仍讓 manifest 測試通過。
    """

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, tzinfo=UTC)
    count = int((end - start) / timedelta(hours=6))
    time_ns = np.array(
        [
            int((start + timedelta(hours=6 * index)).timestamp() * 1_000_000_000)
            for index in range(count)
        ],
        dtype=np.int64,
    )
    hours = np.arange(count, dtype=np.float64) * 6.0
    spring_neap = 1.0 + 0.5 * np.sin(2.0 * np.pi * hours / (24.0 * 14.0))
    elevation = spring_neap * np.sin(2.0 * np.pi * hours / 12.42)
    wave = 1.0 + 0.2 * np.sin(2.0 * np.pi * hours / (24.0 * 7.0))
    current = 0.5 + 0.1 * np.cos(2.0 * np.pi * hours / 12.42)
    wave[500] = 8.0
    current[1_500] = 3.0
    selected = select_arrival_times(
        study_site_id="gongliao",
        time_utc_ns=time_ns,
        elevation_m=elevation,
        significant_wave_height_m=wave,
        current_speed_mps=current,
        valid_forcing=np.ones(count, dtype=bool),
        backward_window_available=np.ones(count, dtype=bool),
        design_version="design_baseline_v3_non_rising_a_v3_local20_20260909",
    )
    payload = _arrival_payload(per_site=1)
    payload["status"] = "generated"
    payload["selection_method_id"] = "select_arrival_times"
    payload["records"] = [asdict(item) for item in selected]
    path = tmp_path / "selector-arrival.json"
    _write_json(path, payload)
    loaded = load_arrival_time_manifest(path, _config(tmp_path), formal=False)
    assert loaded == tuple(selected)
    assert {item.season for item in loaded} == {"DJF", "MAM", "JJA", "SON"}


def test_bed_residence_formal_loader_recomputes_shared_age_vector_and_paired_a_ids(
    tmp_path: Path,
) -> None:
    """正式 loader 依 seed 重算五站 50-age，並接受龜山 paired event 五欄 identity。"""

    config = _bed_config()
    payload = _bed_arrival_payload(config)
    path = tmp_path / "bed-arrival.json"
    _write_json(path, payload)

    loaded = load_arrival_time_manifest(path, config, formal=True)
    assert len(loaded) == 250
    assert all(
        arrival.time_utc_ns == arrival.metadata["deposition_time_utc_ns"]
        for arrival in loaded
    )
    for site_id in SITES:
        site_records = [arrival for arrival in loaded if arrival.study_site_id == site_id]
        ordered = sorted(
            site_records,
            key=lambda item: (
                item.metadata["observation_time_utc_ns"],
                item.metadata["observation_arrival_time_id"],
            ),
        )
        assert tuple(item.metadata["bed_residence_age_hours"] for item in ordered) == (
            sample_bed_residence_age_hours(
                maximum_age_days=90,
                sample_count=50,
                seed=20260916,
            )
        )
    guishan_events = [
        arrival
        for arrival in loaded
        if arrival.study_site_id == "guishan" and arrival.tide_class == "event"
    ]
    assert len(guishan_events) == 2
    assert all(
        arrival.metadata["shared_A_forcing_policy"] == "gongliao_paired_utc_reference_v1"
        for arrival in guishan_events
    )


@pytest.mark.parametrize("drift", ["age", "seed", "deposition", "schema"])
def test_bed_residence_formal_loader_rejects_age_seed_and_deposition_drift(
    tmp_path: Path, drift: str
) -> None:
    """即使 JSON 重新產生，loader 仍拒絕抽樣 seed、age 或觀測／沉底差異漂移。"""

    config = _bed_config()
    payload = _bed_arrival_payload(config)
    first = payload["records"][0]
    if drift == "age":
        first["metadata"]["bed_residence_age_hours"] += 1
    elif drift == "seed":
        first["metadata"]["bed_residence_sampling_seed"] += 1
    elif drift == "schema":
        payload["schema_version"] = "1.0.0"
    else:
        shifted_deposition_ns = first["metadata"]["deposition_time_utc_ns"] + 3_600_000_000_000
        first["metadata"]["deposition_time_utc_ns"] = shifted_deposition_ns
        first["metadata"]["deposition_time_utc"] = (
            datetime.fromtimestamp(shifted_deposition_ns / 1_000_000_000, tz=UTC)
            .isoformat()
            .replace("+00:00", "Z")
        )
        first["time_utc_ns"] = shifted_deposition_ns
    path = tmp_path / f"bed-arrival-{drift}.json"
    _write_json(path, payload)
    with pytest.raises(ValueError):
        load_arrival_time_manifest(path, config, formal=True)


def test_legacy_config_rejects_bed_residence_arrival_schema(tmp_path: Path) -> None:
    """legacy config 僅接受 1.0.0 arrival，不得讀取 bed-only schema 1.1.0。"""

    payload = _bed_arrival_payload(_bed_config())
    path = tmp_path / "bed-arrival-cross-load.json"
    _write_json(path, payload)
    with pytest.raises(ValueError, match="schema_version"):
        load_arrival_time_manifest(path, _config(tmp_path), formal=True)


@pytest.mark.parametrize("mutation", ["duplicate_core", "duplicate_event"])
def test_arrival_formal_strata_fail_fast(tmp_path: Path, mutation: str) -> None:
    """總數仍為 250 時，重複核心格或事件也不得穿過 48+2 formal gate。"""

    payload = _arrival_payload()
    if mutation == "duplicate_core":
        first = payload["records"][0]
        second = payload["records"][1]
        for field in ("year", "season", "tide_class", "phase_or_event"):
            second[field] = first[field]
    else:
        first_site_rows = payload["records"][:50]
        strong_current = next(
            row for row in first_site_rows if row["phase_or_event"] == "strong_current_event"
        )
        strong_current["phase_or_event"] = "high_wave_event"
    path = tmp_path / "arrival.json"
    _write_json(path, payload)
    with pytest.raises(ValueError, match="arrival formal"):
        load_arrival_time_manifest(path, _config(tmp_path), formal=True)


@pytest.mark.parametrize("component", ["receptor", "arrival"])
def test_component_record_unknown_key_fails_fast(tmp_path: Path, component: str) -> None:
    """receptor 與 arrival record 都執行 exact-key 契約，不忽略拼錯欄位。"""

    payload = _receptor_payload(per_site=1) if component == "receptor" else _arrival_payload(per_site=1)
    payload["records"][0]["unexpected"] = "not-allowed"
    path = tmp_path / f"{component}.json"
    _write_json(path, payload)
    loader = load_receptor_manifest if component == "receptor" else load_arrival_time_manifest
    with pytest.raises(ValueError, match="未知"):
        loader(path, _config(tmp_path))


@pytest.mark.parametrize("component", ["receptor", "arrival"])
def test_component_duplicate_identifier_fails_fast(tmp_path: Path, component: str) -> None:
    """receptor_id 與 arrival_time_id 都必須跨站、跨 record 全案唯一。"""

    payload = _receptor_payload(per_site=1) if component == "receptor" else _arrival_payload(per_site=1)
    id_field = "receptor_id" if component == "receptor" else "arrival_time_id"
    payload["records"][1][id_field] = payload["records"][0][id_field]
    path = tmp_path / f"{component}.json"
    _write_json(path, payload)
    loader = load_receptor_manifest if component == "receptor" else load_arrival_time_manifest
    with pytest.raises(ValueError, match="全案唯一"):
        loader(path, _config(tmp_path))


def test_arrival_bool_time_and_legacy_season_fail_fast(tmp_path: Path) -> None:
    """布林時間與舊式 winter 標籤都不可冒充正式 selector 契約。"""

    path = tmp_path / "arrival.json"
    payload = _arrival_payload(per_site=1)
    payload["records"][0]["time_utc_ns"] = True
    _write_json(path, payload)
    with pytest.raises(ValueError, match="不可為 bool"):
        load_arrival_time_manifest(path, _config(tmp_path))
    payload = _arrival_payload(per_site=1)
    payload["records"][0]["season"] = "winter"
    _write_json(path, payload)
    with pytest.raises(ValueError, match="season 不合法"):
        load_arrival_time_manifest(path, _config(tmp_path))


def test_receptor_nonfinite_coordinate_fails_fast(tmp_path: Path) -> None:
    """receptor 經度中的 Infinity 必須在嚴格 JSON 階段拒絕。"""

    payload = _receptor_payload(per_site=1)
    payload["records"][0]["lon"] = float("inf")
    path = tmp_path / "receptor.json"
    _write_json(path, payload)
    with pytest.raises(ValueError, match="非有限常數"):
        load_receptor_manifest(path, _config(tmp_path))


def test_full_scenario_inputs_are_deterministic_and_hash_bound(tmp_path: Path) -> None:
    """正式三份 component 交叉應快速產生 50,000 個 stable scenarios 與三組 hash。"""

    config = _config(tmp_path, with_paths=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true\n", encoding="utf-8")
    _write_json(tmp_path / "material.json", _material_payload())
    _write_json(tmp_path / "receptor.json", _receptor_payload())
    _write_json(tmp_path / "arrival.json", _arrival_payload())
    _write_json(tmp_path / "initial_conditions.json", _initial_condition_payload(config))
    first = load_scenario_inputs(config, config_path=config_path, formal=True)
    second = load_scenario_inputs(config, config_path=config_path, formal=True)
    assert len(first.scenarios) == 50_000
    assert len(first.initial_conditions) == 5_000
    assert len(first.initial_conditions_by_pair) == 5_000
    assert first.scenarios == second.scenarios
    assert set(first.file_sha256) == {
        "material",
        "receptor",
        "arrival",
        "receptor_arrival_initial_condition",
    }
    assert first.canonical_component_hashes == second.canonical_component_hashes
    assert first.records["materials"] == first.materials
    assert first.records["initial_conditions"] == first.initial_conditions
    assert first.initial_conditions[0] == first.initial_conditions_by_pair[
        (first.initial_conditions[0].receptor_id, first.initial_conditions[0].arrival_time_id)
    ]
    same_receptor = [
        item
        for item in first.initial_conditions
        if item.receptor_id == first.initial_conditions[0].receptor_id
    ]
    assert len({item.z_m_positive_up for item in same_receptor}) > 1
    with pytest.raises(TypeError):
        first.file_sha256["extra"] = "b" * 64  # type: ignore[index]
    with pytest.raises(TypeError):
        first.initial_conditions_by_pair[("x", "y")] = first.initial_conditions[0]  # type: ignore[index]


def test_dynamic_initial_condition_resolves_formal_a_domain_and_pilot_base(tmp_path: Path) -> None:
    """legacy expanded policy 的 formal A 區使用 expanded ID，pilot 維持 base。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["design_version"] = "design_baseline_v2_non_rising_oca_proxy"
    payload["scenarios"].pop("bed_residence_time", None)
    payload["inputs"].pop("backtrack_support_days", None)
    payload["scenarios"]["receptor_arrival_initial_condition_manifest"] = None
    payload["domains"][0]["formal_domain_policy"] = "expanded_domain_v1"
    payload["domains"][0].pop("runtime_spatial_support_policy", None)
    payload["domains"][0]["formal_release_flow_domain_id"] = (
        "northeast_taiwan_common_cache_v4_lbt_south_expanded"
    )
    payload["domains"][0]["formal_release_domain_status"] = "approved"
    payload["domains"][0]["expanded_domain_candidate_id"] = (
        "northeast_taiwan_common_cache_v4_lbt_south_expanded"
    )
    payload["domains"][0]["expanded_bbox_lon_lat"] = [121.306315, 122.793685, 24.480000, 25.499156]
    for site in payload["study_sites"]:
        if site["analysis_region_id"] == "A":
            site["formal_release_flow_domain_id"] = (
                "northeast_taiwan_common_cache_v4_lbt_south_expanded"
            )
    config = ProjectConfig.model_validate(payload)
    receptor_payload = _receptor_payload()
    arrival_payload = _arrival_payload()
    receptors = tuple(
        Receptor(
            receptor_id=str(row["receptor_id"]),
            study_site_id=str(row["study_site_id"]),
            analysis_region_id=str(row["analysis_region_id"]),
            lon=float(row["lon"]),
            lat=float(row["lat"]),
            z_m_positive_up=float(row["z_m_positive_up"]),
            vertical_id=str(row["vertical_id"]),
            metadata=row["metadata"],
        )
        for row in receptor_payload["records"]
        if isinstance(row, dict)
    )
    arrivals = tuple(
        ArrivalTime(
            arrival_time_id=str(row["arrival_time_id"]),
            study_site_id=str(row["study_site_id"]),
            time_utc_ns=int(row["time_utc_ns"]),
            year=int(row["year"]),
            season=str(row["season"]),
            tide_class=str(row["tide_class"]),
            phase_or_event=str(row["phase_or_event"]),
            metadata=row["metadata"],
        )
        for row in arrival_payload["records"]
        if isinstance(row, dict)
    )
    formal_payload = _initial_condition_payload(
        config, receptors=receptor_payload, arrivals=arrival_payload, formal=True
    )
    formal_path = tmp_path / "formal-initial.json"
    _write_json(formal_path, formal_payload)
    loaded = load_receptor_arrival_initial_condition_manifest(
        formal_path, config, receptors, arrivals, formal=True
    )
    flow_by_region = {item.analysis_region_id: item.flow_domain_id for item in loaded}
    assert flow_by_region["A"] == "northeast_taiwan_common_cache_v4_lbt_south_expanded"
    assert flow_by_region["B"] == "hsinchu_cache_v3"

    base_payload = deepcopy(formal_payload)
    base_payload["records"][0]["flow_domain_id"] = "northeast_taiwan_common_cache_v3"
    base_path = tmp_path / "base-in-formal.json"
    _write_json(base_path, base_payload)
    with pytest.raises(ValueError, match="resolver"):
        load_receptor_arrival_initial_condition_manifest(
            base_path, config, receptors, arrivals, formal=True
        )

    pilot_receptor_payload = _receptor_payload(per_site=1)
    pilot_arrival_payload = _arrival_payload(per_site=1)
    pilot_receptors = tuple(
        receptor
        for receptor in receptors
        if receptor.receptor_id.endswith("r00")
    )
    pilot_arrivals = tuple(arrival for arrival in arrivals if arrival.arrival_time_id.endswith("t00"))
    pilot_payload = _initial_condition_payload(
        config,
        receptors=pilot_receptor_payload,
        arrivals=pilot_arrival_payload,
        status="pilot",
        formal=False,
    )
    pilot_path = tmp_path / "pilot-initial.json"
    _write_json(pilot_path, pilot_payload)
    pilot_loaded = load_receptor_arrival_initial_condition_manifest(
        pilot_path, config, pilot_receptors, pilot_arrivals, formal=False
    )
    assert {item.flow_domain_id for item in pilot_loaded if item.analysis_region_id == "A"} == {
        "northeast_taiwan_common_cache_v3"
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("receptor_id", "unknown-receptor"),
        ("arrival_time_id", "unknown-arrival"),
        ("study_site_id", "wrong-site"),
        ("analysis_region_id", "wrong-region"),
        ("flow_domain_id", "wrong-flow"),
        ("time_utc_ns", 1),
        ("vertical_id", "wrong-vertical"),
    ],
)
def test_dynamic_initial_condition_rejects_pair_cross_reference(
    tmp_path: Path, field: str, value: object
) -> None:
    """dynamic pair 的 receptor、arrival、站點、region、時間與 vertical 必須逐欄一致。"""

    config, receptors, arrivals, path, payload = _pilot_dynamic_fixture(tmp_path)
    changed = deepcopy(payload)
    changed["records"][0][field] = value
    _write_json(path, changed)
    with pytest.raises(ValueError):
        load_receptor_arrival_initial_condition_manifest(path, config, receptors, arrivals)


def test_dynamic_initial_condition_rejects_missing_duplicate_and_unknown_pairs(tmp_path: Path) -> None:
    """pair 必須恰好一次，缺列、重列與未知 pair 均不可由 row count 掩蓋。"""

    config, receptors, arrivals, path, payload = _pilot_dynamic_fixture(tmp_path)
    missing = deepcopy(payload)
    missing["records"].pop()
    _write_json(path, missing)
    with pytest.raises(ValueError, match="coverage"):
        load_receptor_arrival_initial_condition_manifest(path, config, receptors, arrivals)

    duplicate = deepcopy(payload)
    duplicate["records"].append(deepcopy(duplicate["records"][0]))
    _write_json(path, duplicate)
    with pytest.raises(ValueError, match="不可重複"):
        load_receptor_arrival_initial_condition_manifest(path, config, receptors, arrivals)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wetdry_elem_value", 1),
        ("wetdry_semantics_id", "other_semantics"),
        ("ocm_time_origin", "reconstructed"),
        ("ocm_month_yyyymm", "209901"),
        ("source_face_local_index", -1),
        ("source_face_global_index", -1),
        ("ocm_source_time_index", -1),
    ],
)
def test_dynamic_initial_condition_rejects_source_and_wetdry_contract(
    tmp_path: Path, field: str, value: object
) -> None:
    """dry element、未核准時間來源、月份與負 source index 必須停止。"""

    config, receptors, arrivals, path, payload = _pilot_dynamic_fixture(tmp_path)
    changed = deepcopy(payload)
    changed["records"][0][field] = value
    _write_json(path, changed)
    with pytest.raises(ValueError):
        load_receptor_arrival_initial_condition_manifest(path, config, receptors, arrivals)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eta_m_positive_up", -20.0),
        ("water_column_height_m", 21.000001),
        ("z_m_positive_up", -21.0),
        ("height_above_bed_m", -1.0),
        ("zcor_lower_m_positive_up", 0.0),
        ("zcor_upper_m_positive_up", -19.0),
        ("zcor_lower_m_positive_up", -20.000001),
        ("zcor_upper_m_positive_up", 1.000001),
        ("vertical_bracket_alpha", 1.1),
        ("vertical_bracket_alpha", 0.2),
    ],
)
def test_dynamic_initial_condition_rejects_physical_and_bracket_corruption(
    tmp_path: Path, field: str, value: object
) -> None:
    """eta/bed、z、水柱、離床高與 zcor bracket 的不等式／等式均受物理 gate 保護。"""

    config, receptors, arrivals, path, payload = _pilot_dynamic_fixture(tmp_path)
    changed = deepcopy(payload)
    changed["records"][0][field] = value
    _write_json(path, changed)
    with pytest.raises(ValueError):
        load_receptor_arrival_initial_condition_manifest(path, config, receptors, arrivals)


def test_dynamic_initial_condition_allows_only_serialization_tolerance(tmp_path: Path) -> None:
    """1e-8 內的 JSON 浮點尾差可接受，但不放寬物理範圍。"""

    config, receptors, arrivals, path, payload = _pilot_dynamic_fixture(tmp_path)
    changed = deepcopy(payload)
    changed["records"][0]["water_column_height_m"] += 0.5e-8
    changed["records"][0]["zcor_lower_m_positive_up"] -= 0.5e-8
    changed["records"][0]["zcor_upper_m_positive_up"] += 0.5e-8
    changed["records"][0]["vertical_bracket_alpha"] += 0.5e-8
    _write_json(path, changed)
    loaded = load_receptor_arrival_initial_condition_manifest(path, config, receptors, arrivals)
    assert len(loaded) == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "pilot"),
        ("manifest_kind", "receptor_manifest"),
        ("schema_version", "2.0.0"),
        ("vertical_reference", "z_positive_down"),
        ("time_standard", "local"),
        ("provenance", {"method_id": "only"}),
    ],
)
def test_dynamic_initial_condition_root_is_strict_in_formal_mode(
    tmp_path: Path, field: str, value: object
) -> None:
    """formal dynamic manifest 必須 approved、固定 kind/version、UTC 與完整 provenance。"""

    config, receptors, arrivals, path, payload = _pilot_dynamic_fixture(tmp_path)
    changed = deepcopy(payload)
    changed[field] = value
    _write_json(path, changed)
    with pytest.raises(ValueError):
        load_receptor_arrival_initial_condition_manifest(path, config, receptors, arrivals, formal=True)


def test_manifest_relative_path_uses_config_directory_not_cwd(tmp_path: Path) -> None:
    """相對路徑固定以 config YAML parent 為基準，沒有基準時拒絕猜 cwd。"""

    config_path = tmp_path / "nested" / "config.yaml"
    config_path.parent.mkdir()
    assert resolve_manifest_path(config_path, "components/x.json") == (
        config_path.parent / "components/x.json"
    )
    absolute = tmp_path / "absolute.json"
    assert resolve_manifest_path(config_path, absolute) == absolute
    with pytest.raises(ValueError, match="config YAML"):
        resolve_manifest_path(None, "components/x.json")
    with pytest.raises(ValueError, match="空白"):
        resolve_manifest_path(config_path, " ")


def test_dynamic_initial_condition_requirement_and_explicit_fallback_gate(tmp_path: Path) -> None:
    """nonformal legacy 可省略 dynamic；明示 require 或 explicit/fallback 衝突則拒絕。"""

    config = _config(tmp_path)
    material_path = tmp_path / "material.json"
    receptor_path = tmp_path / "receptor.json"
    arrival_path = tmp_path / "arrival.json"
    _write_json(material_path, _material_payload())
    _write_json(receptor_path, _receptor_payload(per_site=1))
    _write_json(arrival_path, _arrival_payload(per_site=1))
    legacy = load_scenario_inputs(
        config,
        material_path=material_path,
        receptor_path=receptor_path,
        arrival_path=arrival_path,
        formal=False,
    )
    assert legacy.initial_conditions == ()
    with pytest.raises(ValueError, match="receptor_arrival_initial_condition manifest path"):
        load_scenario_inputs(
            config,
            material_path=material_path,
            receptor_path=receptor_path,
            arrival_path=arrival_path,
            require_dynamic_initial_conditions=True,
            formal=False,
        )

    configured = _config(tmp_path / "configured", with_paths=True)
    config_path = tmp_path / "configured.yaml"
    with pytest.raises(ValueError, match="explicit path"):
        load_scenario_inputs(
            configured,
            config_path=config_path,
            initial_condition_path="another-initial-conditions.json",
            formal=False,
        )


def test_formal_geometry_builds_metric_nested_boundaries_and_foreign_domains(tmp_path: Path) -> None:
    """四域五站 fixture 應投影成公尺、A 區保留 foreign local，其餘站點 local=flow。"""

    config = _config(tmp_path)
    domain, local, opened = _geometry_payloads()
    domain_path, local_path, open_path = (
        tmp_path / name for name in ("domain.json", "local.json", "open.json")
    )
    _write_json(domain_path, domain)
    _write_json(local_path, local)
    _write_json(open_path, opened)
    bundle = load_boundary_geometries(
        config,
        domain_path,
        local_path,
        open_path,
        formal=True,
    )
    assert set(bundle) == set(SITES)
    assert set(bundle["gongliao"].foreign_local_domains) == {"guishan"}
    assert bundle["guishan"].foreign_local_domains == {"gongliao": bundle["gongliao"].own_local_domain}
    assert bundle["hsinchu"].local_equals_flow
    assert bundle["hsinchu"].own_local_domain.equals(bundle["hsinchu"].flow_domain)
    assert bundle["hsinchu"].own_local_open_boundary.equals(bundle["hsinchu"].flow_open_boundary)
    assert bundle["gongliao"].flow_domain.area > 1_000_000.0
    assert bundle.canonical_component_hashes["domain"]


def test_geometry_formal_uses_resolved_a_v4_and_pilot_keeps_base_id(tmp_path: Path) -> None:
    """legacy expanded policy 的 formal A 只接受設定解析出的 v4；pilot 維持 base v3。"""

    formal_flow = "northeast_taiwan_common_cache_v4_lbt_south_expanded"
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["design_version"] = "design_baseline_v2_non_rising_oca_proxy"
    payload["scenarios"].pop("bed_residence_time", None)
    payload["inputs"].pop("backtrack_support_days", None)
    payload["domains"][0]["formal_domain_policy"] = "expanded_domain_v1"
    payload["domains"][0].pop("runtime_spatial_support_policy", None)
    payload["domains"][0]["formal_release_flow_domain_id"] = formal_flow
    payload["domains"][0]["formal_release_domain_status"] = "approved"
    payload["domains"][0]["expanded_domain_candidate_id"] = formal_flow
    payload["domains"][0]["expanded_bbox_lon_lat"] = [121.306315, 122.793685, 24.480000, 25.499156]
    for site in payload["study_sites"]:
        if site["analysis_region_id"] == "A":
            site["formal_release_flow_domain_id"] = formal_flow
    config = ProjectConfig.model_validate(payload)

    formal_domain, formal_local, formal_open = _geometry_payloads(a_flow_id=formal_flow)
    for geometry_payload in (formal_domain, formal_local, formal_open):
        geometry_payload["design_version"] = config.design_version
    formal_paths = [
        tmp_path / name
        for name in ("formal-domain.json", "formal-local.json", "formal-open.json")
    ]
    for path, value in zip(formal_paths, (formal_domain, formal_local, formal_open), strict=True):
        _write_json(path, value)
    formal_bundle = load_boundary_geometries(config, *formal_paths, formal=True)
    assert formal_bundle["gongliao"].flow_boundary_segment_id.startswith(f"{formal_flow}_")

    base_domain, base_local, base_open = _geometry_payloads()
    for geometry_payload in (base_domain, base_local, base_open):
        geometry_payload["design_version"] = config.design_version
    base_paths = [tmp_path / name for name in ("base-domain.json", "base-local.json", "base-open.json")]
    for path, value in zip(base_paths, (base_domain, base_local, base_open), strict=True):
        _write_json(path, value)
    with pytest.raises(ValueError, match="resolver|config"):
        load_boundary_geometries(config, *base_paths, formal=True)

    pilot_bundle = load_boundary_geometries(config, *base_paths, formal=False)
    assert pilot_bundle["gongliao"].flow_boundary_segment_id.startswith(
        "northeast_taiwan_common_cache_v3_"
    )


def test_geometry_pilot_allows_selected_flow_subset_but_rejects_extra_domain(
    tmp_path: Path,
) -> None:
    """pilot 可只載入 A 區兩站，但不得帶入未被站點使用的 B 域。"""

    config = _config(tmp_path)
    domain, local, opened = _geometry_payloads()
    domain["records"] = [
        record for record in domain["records"] if record["analysis_region_id"] == "A"
    ]
    local["records"] = [
        record for record in local["records"] if record["analysis_region_id"] == "A"
    ]
    opened["records"] = [
        record for record in opened["records"] if record["analysis_region_id"] == "A"
    ]
    paths = [tmp_path / name for name in ("pilot-domain.json", "pilot-local.json", "pilot-open.json")]
    for path, payload in zip(paths, (domain, local, opened), strict=True):
        _write_json(path, payload)
    bundle = load_boundary_geometries(config, *paths, formal=False)
    assert set(bundle) == {"gongliao", "guishan"}

    full_domain, _, _ = _geometry_payloads()
    domain_with_extra = deepcopy(domain)
    domain_with_extra["records"].append(
        next(record for record in full_domain["records"] if record["analysis_region_id"] == "B")
    )
    _write_json(paths[0], domain_with_extra)
    with pytest.raises(ValueError, match="selected sites|非空子集"):
        load_boundary_geometries(config, *paths, formal=False)

    unknown_domain = deepcopy(domain)
    unknown_domain["records"][0]["flow_domain_id"] = "unknown-flow-domain"
    _write_json(paths[0], unknown_domain)
    with pytest.raises(ValueError, match="resolver|config"):
        load_boundary_geometries(config, *paths, formal=False)


@pytest.mark.parametrize("mutation", ["missing_local_line", "off_boundary", "wrong_crs", "wrong_type"])
def test_geometry_boundary_contract_rejects_invalid_inputs(tmp_path: Path, mutation: str) -> None:
    """local/open boundary 缺漏、偏離 polygon、CRS 或 geometry type 錯誤必須 fail-fast。"""

    config = _config(tmp_path)
    domain, local, opened = _geometry_payloads()
    if mutation == "missing_local_line":
        opened["records"] = [
            item for item in opened["records"] if item["owner_kind"] != "local_domain"
        ]
    elif mutation == "off_boundary":
        for item in opened["records"]:
            if item["owner_kind"] == "local_domain" and item["owner_id"] == "gongliao":
                item["geometry"] = mapping(LineString([(121.8, 25.05), (122.0, 25.05)]))
    elif mutation == "wrong_crs":
        domain["coordinate_reference"] = "EPSG:3857"
    else:
        domain["records"][0]["geometry"] = mapping(LineString([(121.3, 24.6), (122.8, 24.6)]))
    paths = [tmp_path / name for name in ("domain.json", "local.json", "open.json")]
    for path, payload in zip(paths, (domain, local, opened), strict=True):
        _write_json(path, payload)
    with pytest.raises(ValueError):
        load_boundary_geometries(config, *paths, formal=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("out_of_range", "WGS84"),
        ("bool", "不可為 bool"),
        ("nonfinite", "非有限常數"),
        ("nonnumeric", "有限數值"),
        ("mixed_nesting", "二維"),
    ],
)
def test_geojson_coordinates_are_strict_before_shapely(
    tmp_path: Path, mutation: str, message: str
) -> None:
    """GeoJSON 的範圍、型別、有限性與 nesting 必須在 Shapely 解析前 fail-fast。"""

    config = _config(tmp_path)
    domain, local, opened = _geometry_payloads()
    geometry = domain["records"][0]["geometry"]
    coordinates = [list(position) for position in geometry["coordinates"][0]]
    if mutation == "out_of_range":
        coordinates[0][0] = 181.0
    elif mutation == "bool":
        coordinates[0][0] = True
    elif mutation == "nonfinite":
        coordinates[0][0] = float("nan")
    elif mutation == "nonnumeric":
        coordinates[0][0] = "121.3"
    else:
        coordinates[0] = [coordinates[0]]
    geometry["coordinates"] = [coordinates]
    paths = [tmp_path / name for name in ("domain.json", "local.json", "open.json")]
    for path, payload in zip(paths, (domain, local, opened), strict=True):
        _write_json(path, payload)
    with pytest.raises(ValueError, match=message):
        load_boundary_geometries(config, *paths, formal=True)

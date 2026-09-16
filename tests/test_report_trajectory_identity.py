"""報告代表軌跡 identity、狀態閘門與 defensive immutable record 測試。

本檔只建立小型 synthetic ``ParticleResult`` 與 ``Observation``，驗證 canonical JSON
SHA-256、原生型別限制、報告有效成員政策及代表軌跡的 tuple/frozen 行為。測試資料使用
既有公尺、秒與 UTC 奈秒欄位，但不代表 OCM／NWW3 正式科學產品或絕對來源機率。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from lagrangian_backtracking.engine import EnvironmentSampleStatus, Observation, ParticleResult
from lagrangian_backtracking.models import ParticleState, ParticleStatus, VelocitySampleStatus
from lagrangian_backtracking.report_trajectory_identity import (
    CORE_SEASONS,
    CORE_TIDE_CLASSES,
    REPRESENTATIVE_SELECTION_POLICY,
    RepresentativeTrajectory,
    is_valid_report_member,
    representative_priority_digest,
)

_IDENTITY = {
    "particle_id": "particle-0",
    "scenario_id": "scenario-0",
    "member_id": 0,
    "study_site_id": "site-0",
    "analysis_region_id": "region-0",
    "receptor_id": "receptor-0",
    "material_id": "material-0",
    "arrival_time_id": "arrival-0",
    "season": "DJF",
    "tide_class": "spring_proxy",
}


def _digest(**overrides: object) -> str:
    """用固定 identity 建立摘要 fixture，讓每個欄位變化測試保持獨立。"""

    values: dict[str, object] = {"selection_seed": 7, **_IDENTITY}
    values.update(overrides)
    return representative_priority_digest(**values)  # type: ignore[arg-type]


def _state(status: ParticleStatus) -> ParticleState:
    """建立只含 identity 與 terminal status 的最小粒子狀態。"""

    return ParticleState(
        particle_id="particle-0",
        scenario_id="scenario-0",
        member_id=0,
        study_site_id="site-0",
        analysis_region_id="region-0",
        receptor_id="receptor-0",
        x_m=1.0,
        y_m=2.0,
        z_m=-3.0,
        time_utc_ns=10,
        age_seconds=1.0,
        status=status,
    )


def _result(status: ParticleStatus) -> ParticleResult:
    """把 synthetic terminal state 包成 exact ``ParticleResult``。"""

    return ParticleResult(
        final_state=_state(status),
        observations=[],
        events=[],
        step_count=0,
        minimum_clamp_count=0,
    )


def _observations() -> list[Observation]:
    """建立同一粒子、兩個時間點的最小 backward observation 序列。"""

    return [
        Observation("particle-0", 20, 0.0, 0.0, 0.0, -1.0, ParticleStatus.ACTIVE),
        Observation("particle-0", 10, 1.0, 1.0, 2.0, -3.0, ParticleStatus.MAX_AGE),
    ]


def _observations_with_environment_context() -> list[Observation]:
    """建立帶有環境與同點逐步速度 context 的 snapshot fixture。

    兩個觀測都在同一個有限公尺垂向範圍內，讓測試能確認 representative snapshot 會
    保留環境上下文與九個速度欄位；這些 synthetic 欄位只測試資料契約，不代表真實
    OCM／NWW3 樣本。
    """

    return [
        Observation(
            "particle-0",
            20,
            0.0,
            0.0,
            0.0,
            -1.0,
            ParticleStatus.ACTIVE,
            EnvironmentSampleStatus.VALID,
            2.0,
            -10.0,
            "202401",
            0,
            velocity_sample_status=VelocitySampleStatus.COMPLETE,
            total_u_mps=3.0,
            total_v_mps=-1.0,
            total_w_mps=-0.5,
            ocm_u_mps=2.5,
            ocm_v_mps=-1.2,
            ocm_w_mps=-0.25,
            stokes_u_mps=0.5,
            stokes_v_mps=0.2,
            settling_w_mps=-0.25,
            velocity_qc_flags=0,
        ),
        Observation(
            "particle-0",
            10,
            1.0,
            1.0,
            2.0,
            -3.0,
            ParticleStatus.MAX_AGE,
            EnvironmentSampleStatus.VALID,
            2.0,
            -10.0,
            "202401",
            0,
            velocity_sample_status=VelocitySampleStatus.COMPLETE,
            total_u_mps=3.0,
            total_v_mps=-1.0,
            total_w_mps=-0.5,
            ocm_u_mps=2.5,
            ocm_v_mps=-1.2,
            ocm_w_mps=-0.25,
            stokes_u_mps=0.5,
            stokes_v_mps=0.2,
            settling_w_mps=-0.25,
            velocity_qc_flags=0,
        ),
    ]


def test_core_strata_and_policy_are_fixed() -> None:
    """核心報告固定四季、兩潮況與版本化選樣 policy。"""

    assert CORE_SEASONS == ("DJF", "MAM", "JJA", "SON")
    assert CORE_TIDE_CLASSES == ("spring_proxy", "neap_proxy")
    assert REPRESENTATIVE_SELECTION_POLICY == "stable_hash_core_season_tide_v1"


def test_priority_digest_matches_sorted_compact_utf8_json_and_is_deterministic() -> None:
    """同一資料不受 keyword 順序影響，並精確符合 compact UTF-8 canonical JSON。"""

    digest = representative_priority_digest(
        selection_seed=7,
        receptor_id="受體-0",
        analysis_region_id="region-0",
        member_id=0,
        study_site_id="site-0",
        scenario_id="scenario-0",
        particle_id="particle-0",
        material_id="material-0",
        arrival_time_id="arrival-0",
        season="DJF",
        tide_class="spring_proxy",
    )
    payload = {
        "identity": {
            "particle_id": "particle-0",
            "scenario_id": "scenario-0",
            "member_id": 0,
            "study_site_id": "site-0",
            "analysis_region_id": "region-0",
            "receptor_id": "受體-0",
            "material_id": "material-0",
            "arrival_time_id": "arrival-0",
            "season": "DJF",
            "tide_class": "spring_proxy",
        },
        "selection_policy": REPRESENTATIVE_SELECTION_POLICY,
        "selection_seed": "7",
    }
    expected = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert digest == expected
    assert digest == representative_priority_digest(
        7,
        "particle-0",
        "scenario-0",
        0,
        "site-0",
        "region-0",
        "受體-0",
        "material-0",
        "arrival-0",
        "DJF",
        "spring_proxy",
    )
    assert len(digest) == 64
    assert digest == digest.lower()


@pytest.mark.parametrize(
    "field,value",
    [
        ("selection_seed", 8),
        ("particle_id", "particle-1"),
        ("scenario_id", "scenario-1"),
        ("member_id", 1),
        ("study_site_id", "site-1"),
        ("analysis_region_id", "region-1"),
        ("receptor_id", "receptor-1"),
        ("material_id", "material-1"),
        ("arrival_time_id", "arrival-1"),
        ("season", "MAM"),
        ("tide_class", "neap_proxy"),
    ],
)
def test_any_identity_or_seed_change_changes_digest(field: str, value: object) -> None:
    """seed 或十個 identity 任一欄位變動都必須改變優先序摘要。"""

    original = _digest()
    changed = _digest(**{field: value})
    assert changed != original


@pytest.mark.parametrize(
    "seed,expected_exception",
    [
        (True, TypeError),
        (np.int64(0), TypeError),
        (-1, ValueError),
        (2**128, ValueError),
    ],
)
def test_digest_seed_type_and_range_gates(seed: object, expected_exception: type[Exception]) -> None:
    """selection seed 嚴格限制原生型別與 128 位元無號範圍。"""

    with pytest.raises(expected_exception):
        _digest(selection_seed=seed)


@pytest.mark.parametrize(
    "field,value,expected_exception",
    [
        ("particle_id", "", ValueError),
        ("scenario_id", " scenario-0", ValueError),
        ("member_id", True, TypeError),
        ("member_id", np.int64(0), TypeError),
        ("member_id", -1, ValueError),
        ("study_site_id", np.str_("site-0"), TypeError),
        ("analysis_region_id", "", ValueError),
        ("receptor_id", None, TypeError),
        ("material_id", "", ValueError),
        ("arrival_time_id", np.str_("arrival-0"), TypeError),
        ("season", "MAY", ValueError),
        ("season", np.str_("DJF"), TypeError),
        ("tide_class", "event", ValueError),
        ("tide_class", None, TypeError),
    ],
)
def test_digest_identity_type_and_nonempty_gates(
    field: str,
    value: object,
    expected_exception: type[Exception],
) -> None:
    """完整 identity 每欄都必須通過原生型別與非空驗證。"""

    with pytest.raises(expected_exception):
        _digest(**{field: value})


def test_report_member_status_policy() -> None:
    """ACTIVE 拒絕；兩種計算失敗與 pre-window 均排除；其餘終止狀態有效。"""

    with pytest.raises(ValueError):
        is_valid_report_member(_result(ParticleStatus.ACTIVE))
    assert is_valid_report_member(_result(ParticleStatus.DATA_GAP)) is False
    assert is_valid_report_member(_result(ParticleStatus.NUMERICAL_FAILURE)) is False
    assert is_valid_report_member(_result(ParticleStatus.PRE_WINDOW_DEPOSITION)) is False
    terminal_statuses = tuple(
        status
        for status in ParticleStatus
        if status
        not in {
            ParticleStatus.ACTIVE,
            ParticleStatus.DATA_GAP,
            ParticleStatus.NUMERICAL_FAILURE,
            ParticleStatus.PRE_WINDOW_DEPOSITION,
        }
    )
    assert terminal_statuses
    assert all(is_valid_report_member(_result(status)) for status in terminal_statuses)


def test_report_member_requires_exact_particle_result_and_state_contract() -> None:
    """status policy 不接受 duck-typed object、子類或錯誤 final state/status 型別。"""

    class DerivedParticleResult(ParticleResult):
        """只供 exact-type gate 測試的子類。"""

    derived = DerivedParticleResult(
        final_state=_state(ParticleStatus.MAX_AGE),
        observations=[],
        events=[],
        step_count=0,
        minimum_clamp_count=0,
    )
    with pytest.raises(TypeError):
        is_valid_report_member(derived)
    with pytest.raises(TypeError):
        is_valid_report_member(object())  # type: ignore[arg-type]

    malformed = _result(ParticleStatus.MAX_AGE)
    malformed.final_state = None  # type: ignore[assignment]
    with pytest.raises(TypeError):
        is_valid_report_member(malformed)


def test_representative_trajectory_defensively_snapshots_and_is_frozen() -> None:
    """輸入 list 變更不污染 record，觀測為 tuple，欄位也不能重新賦值。"""

    observations = _observations()
    record = RepresentativeTrajectory(
        **_IDENTITY,
        priority_digest="a" * 64,
        observations=observations,
    )
    observations.append(Observation("particle-0", 0, 2.0, 2.0, 3.0, -4.0, ParticleStatus.MAX_AGE))
    assert isinstance(record.observations, tuple)
    assert len(record.observations) == 2
    assert record.digest == record.priority_digest == "a" * 64
    assert record.particle_id == _IDENTITY["particle_id"]
    assert record.material_id == _IDENTITY["material_id"]
    assert record.arrival_time_id == _IDENTITY["arrival_time_id"]
    assert record.season == _IDENTITY["season"]
    assert record.tide_class == _IDENTITY["tide_class"]
    with pytest.raises(FrozenInstanceError):
        record.receptor_id = "other"  # type: ignore[misc]


def test_representative_trajectory_rebuilds_observations_and_preserves_context() -> None:
    """snapshot 必須建立新 Observation，並完整保留環境與速度 context。"""

    source = _observations_with_environment_context()
    record = RepresentativeTrajectory(
        **_IDENTITY,
        priority_digest="a" * 64,
        observations=source,
    )

    assert record.observations is not source
    assert all(
        saved is not original
        for saved, original in zip(record.observations, source, strict=True)
    )
    for saved, original in zip(record.observations, source, strict=True):
        assert saved == original
        assert saved.environment_sample_status is original.environment_sample_status
        assert saved.eta_m == original.eta_m
        assert saved.bed_z_m == original.bed_z_m
        assert saved.forcing_month_id == original.forcing_month_id
        assert saved.environment_qc_flags == original.environment_qc_flags
        assert saved.velocity_sample_status is original.velocity_sample_status
        assert saved.total_u_mps == original.total_u_mps
        assert saved.total_v_mps == original.total_v_mps
        assert saved.total_w_mps == original.total_w_mps
        assert saved.ocm_u_mps == original.ocm_u_mps
        assert saved.ocm_v_mps == original.ocm_v_mps
        assert saved.ocm_w_mps == original.ocm_w_mps
        assert saved.stokes_u_mps == original.stokes_u_mps
        assert saved.stokes_v_mps == original.stokes_v_mps
        assert saved.settling_w_mps == original.settling_w_mps
        assert saved.velocity_qc_flags == original.velocity_qc_flags

    # 即使 caller 以低階方式竄改原始 frozen object，record 仍只能看到自己的 canonical copy。
    object.__setattr__(source[0], "x_m", 999.0)
    assert record.observations[0].x_m == 0.0


@pytest.mark.parametrize("field", ["age_seconds", "x_m", "y_m", "z_m"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_representative_trajectory_rejects_nonfinite_age_or_coordinates(
    field: str,
    value: float,
) -> None:
    """年齡與公尺座標必須是有限值，避免不可畫出的軌跡進入報告。"""

    observations = _observations()
    object.__setattr__(observations[0], field, value)
    with pytest.raises(ValueError):
        RepresentativeTrajectory(
            **_IDENTITY,
            priority_digest="a" * 64,
            observations=observations,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("time_utc_ns", True),
        ("time_utc_ns", np.int64(20)),
        ("age_seconds", 0),
        ("age_seconds", True),
        ("age_seconds", np.float64(0.0)),
        ("x_m", 0),
        ("x_m", False),
        ("x_m", np.float64(0.0)),
        ("y_m", 0),
        ("z_m", np.float64(0.0)),
    ],
)
def test_representative_trajectory_rejects_non_native_observation_scalars(
    field: str,
    value: object,
) -> None:
    """時間拒絕 bool／NumPy 整數，公尺與秒欄位拒絕 int／bool／NumPy scalar。"""

    observations = _observations()
    object.__setattr__(observations[0], field, value)
    with pytest.raises(TypeError):
        RepresentativeTrajectory(
            **_IDENTITY,
            priority_digest="a" * 64,
            observations=observations,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("age_seconds", 0.0),
        ("age_seconds", -1.0),
    ],
)
def test_representative_trajectory_requires_strictly_increasing_age(
    field: str,
    value: float,
) -> None:
    """backtracking age 必須沿觀測序列嚴格增加且不得倒退。"""

    observations = _observations()
    object.__setattr__(observations[1], field, value)
    with pytest.raises(ValueError):
        RepresentativeTrajectory(
            **_IDENTITY,
            priority_digest="a" * 64,
            observations=observations,
        )


def test_representative_trajectory_rejects_negative_age() -> None:
    """回溯年齡以秒表示且不得小於零，即使序列本身仍維持遞增也要拒絕。"""

    observations = _observations()
    object.__setattr__(observations[0], "age_seconds", -0.5)
    object.__setattr__(observations[1], "age_seconds", 0.5)
    with pytest.raises(ValueError):
        RepresentativeTrajectory(
            **_IDENTITY,
            priority_digest="a" * 64,
            observations=observations,
        )


@pytest.mark.parametrize("time_utc_ns", [20, 30])
def test_representative_trajectory_requires_strictly_decreasing_utc_time(
    time_utc_ns: int,
) -> None:
    """逆向軌跡的 UTC 奈秒必須嚴格遞減，不接受重複或向未來前進。"""

    observations = _observations()
    object.__setattr__(observations[1], "time_utc_ns", time_utc_ns)
    with pytest.raises(ValueError):
        RepresentativeTrajectory(
            **_IDENTITY,
            priority_digest="a" * 64,
            observations=observations,
        )


def test_representative_trajectory_rejects_middle_terminal_observation() -> None:
    """完整軌跡不能在最後一筆之前出現 terminal 狀態。"""

    observations = _observations()
    observations.append(
        Observation("particle-0", 0, 2.0, 2.0, 3.0, -4.0, ParticleStatus.MAX_AGE)
    )
    with pytest.raises(ValueError):
        RepresentativeTrajectory(
            **_IDENTITY,
            priority_digest="a" * 64,
            observations=observations,
        )


def test_representative_trajectory_rejects_active_final_observation() -> None:
    """最後一筆仍為 ACTIVE 代表結果尚未完成，不能封存成代表軌跡。"""

    observations = _observations()
    object.__setattr__(observations[-1], "status", ParticleStatus.ACTIVE)
    with pytest.raises(ValueError):
        RepresentativeTrajectory(
            **_IDENTITY,
            priority_digest="a" * 64,
            observations=observations,
        )


def test_representative_trajectory_rejects_particle_id_mismatch() -> None:
    """每筆觀測的原生粒子識別碼都必須與代表 identity 完全相同。"""

    observations = _observations()
    object.__setattr__(observations[1], "particle_id", "other-particle")
    with pytest.raises(ValueError):
        RepresentativeTrajectory(
            **_IDENTITY,
            priority_digest="a" * 64,
            observations=observations,
        )


@pytest.mark.parametrize(
    "field,value,expected_exception",
    [
        ("environment_sample_status", "valid", TypeError),
        ("environment_sample_status", EnvironmentSampleStatus.NOT_SAMPLED, ValueError),
        ("environment_qc_flags", np.int64(0), ValueError),
        ("environment_qc_flags", 1, ValueError),
        ("eta_m", float("nan"), ValueError),
        ("forcing_month_id", "202413", ValueError),
    ],
)
def test_representative_trajectory_revalidates_tampered_environment_context(
    field: str,
    value: object,
    expected_exception: type[Exception],
) -> None:
    """即使來源 Observation 被低階竄改，重建時仍須由既有 context 契約 fail closed。"""

    observations = _observations_with_environment_context()
    object.__setattr__(observations[0], field, value)
    with pytest.raises(expected_exception):
        RepresentativeTrajectory(
            **_IDENTITY,
            priority_digest="a" * 64,
            observations=observations,
        )


@pytest.mark.parametrize(
    "overrides,expected_exception",
    [
        ({"priority_digest": "short"}, ValueError),
        ({"observations": [_observations()[0]]}, ValueError),
        ({"observations": [object(), _observations()[1]]}, TypeError),
        (
            {
                "observations": _observations()[:1]
                + [
                    Observation(
                        "other",
                        0,
                        2.0,
                        2.0,
                        3.0,
                        -4.0,
                        ParticleStatus.MAX_AGE,
                    )
                ]
            },
            ValueError,
        ),
        ({"material_id": ""}, ValueError),
        ({"arrival_time_id": None}, TypeError),
        ({"season": "event"}, ValueError),
        ({"tide_class": "event"}, ValueError),
        ({"member_id": True}, TypeError),
    ],
)
def test_representative_trajectory_contract_gates(
    overrides: dict[str, object],
    expected_exception: type[Exception],
) -> None:
    """代表 record 拒絕錯誤摘要、短軌跡、錯誤觀測與非原生 identity。"""

    values: dict[str, object] = {
        **_IDENTITY,
        "priority_digest": "a" * 64,
        "observations": _observations(),
    }
    values.update(overrides)
    with pytest.raises(expected_exception):
        RepresentativeTrajectory(**values)  # type: ignore[arg-type]

import pytest

from suwon_chat.metrics import display_dist, distance, mean_score, ordinal_emd, pearson, tv_distance


def test_wd_is_zero_for_identical_distributions():
    assert ordinal_emd([0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [1, 2, 3]) == 0


def test_wd_is_one_when_all_mass_moves_end_to_end():
    assert ordinal_emd([1, 0, 0, 0], [0, 0, 0, 1], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_wd_of_one_step_shift_on_seven_point_scale():
    # 전부 4점 → 전부 5점: 한 칸 이동은 척도 폭 6 가운데 1
    p = [0, 0, 0, 1, 0, 0, 0]
    q = [0, 0, 0, 0, 1, 0, 0]
    assert ordinal_emd(p, q, [1, 2, 3, 4, 5, 6, 7]) == pytest.approx(1 / 6)


def test_wd_ignores_refusal_slot_beyond_ordinal_length():
    # 무응답 칸(순서 값보다 긴 꼬리)은 버린다
    assert ordinal_emd([0.5, 0.5, 0.9], [0.5, 0.5, 0.0], [1, 2]) == 0


def test_wd_drops_out_of_scale_option():
    # 순서 값이 -1인 '잘 모르겠다'는 빼고 나머지를 다시 정규화한다
    p = [0.4, 0.4, 0.2]
    q = [0.5, 0.5, 0.0]
    assert ordinal_emd(p, q, [1, 2, -1]) == pytest.approx(0.0)


def test_binary_wd_equals_share_difference():
    assert ordinal_emd([0.3, 0.7], [0.45, 0.55], [1, 2]) == pytest.approx(0.15)


def test_nominal_uses_total_variation():
    p, q = [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]
    assert ordinal_emd(p, q, [1, 1, 1]) == 0          # 순서가 없으면 WD는 뜻이 없다
    assert tv_distance(p, q) == pytest.approx(0.3)
    assert distance(p, q, [1, 1, 1]) == pytest.approx(0.3)


def test_mean_score():
    assert mean_score([0.25, 0.25, 0.5], [1, 2, 3]) == pytest.approx(2.25)
    assert mean_score([0.25, 0.25, 0.5], [1, 2, -1]) == pytest.approx(1.5)
    assert mean_score([0.5, 0.5], [1, 1]) is None


def test_display_dist_drops_refusal_and_renormalises():
    assert display_dist([0.2, 0.2, 0.6], 2) == pytest.approx([0.5, 0.5])


def test_pearson_needs_three_groups():
    assert pearson([1, 2], [1, 2]) is None
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

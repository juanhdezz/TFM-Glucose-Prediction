# -*- coding: utf-8 -*-
"""Tests básicos del loader (ejecutar con pytest)."""

from analysis.loader import parse_condition


def test_parse_original():
    r = parse_condition("original")
    assert r["technique"] == "original"
    assert r["dimension"] is None


def test_parse_balanced_age():
    r = parse_condition("balanced_age_smote")
    assert r["dimension"] == "age"
    assert r["technique"] == "smote"


def test_parse_balanced_sex_compound():
    r = parse_condition("balanced_sex_borderline_smote")
    assert r["dimension"] == "sex"
    assert r["technique"] == "borderline_smote"

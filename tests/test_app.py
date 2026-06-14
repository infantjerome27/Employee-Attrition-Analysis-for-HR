import pandas as pd

from app import load_data


def test_dataset_has_expected_columns_and_attrition_labels():
    df = load_data()

    assert set(["Age", "Salary", "Department", "YearsAtCompany", "Overtime", "JobSatisfaction", "Attrition"]).issubset(df.columns)
    assert len(df) > 0
    assert set(df["Attrition"].unique()).issubset({"Yes", "No"})


def test_attrition_rate_is_between_zero_and_one():
    df = load_data()

    rate = (df["Attrition"] == "Yes").mean()

    assert 0 <= rate <= 1

import pandas as pd


def get_irrelevant_features(
    correlation_df: pd.DataFrame, target: str, threshold: float
) -> list[str]:
    correlated_column_pairs = get_correlated_feature_pairs(correlation_df, threshold)

    drop_cols = []

    for corr_pair in correlated_column_pairs:
        corr0 = get_feature_target_correlation(corr_pair[0], target, correlation_df)
        corr1 = get_feature_target_correlation(corr_pair[1], target, correlation_df)

        if corr0 > corr1:
            drop_cols.append(corr_pair[1])
        else:
            drop_cols.append(corr_pair[0])

    return list(set(drop_cols))


def get_correlated_feature_pairs(
    correlation_df: pd.DataFrame, threshold: float
) -> list[tuple[str, str]]:
    correlated_cols = []

    for i, _ in enumerate(correlation_df.columns):
        for j in range(i):
            if abs(correlation_df.iloc[i, j]) > threshold:
                correlated_cols.append(
                    (correlation_df.columns[i], correlation_df.columns[j])
                )

    return [pair for pair in correlated_cols if "diagnosis" not in pair]  # type: ignore


def get_feature_target_correlation(
    feature: str, target: str, correlation_df: pd.DataFrame
) -> float:
    return correlation_df.loc[feature, target]

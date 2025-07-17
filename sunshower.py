# sunshower.py
import pandas as pd
import json
from typing import List
import openai
import os
import numpy as np
from sklearn.metrics import mutual_info_score

def compute_mutual_info(df):
    mi = {}
    num_cols = df.select_dtypes(include=[np.number]).columns
    for i, col1 in enumerate(num_cols):
        for col2 in num_cols[i+1:]:
            try:
                # Drop NA and align indices
                s1 = df[col1].dropna()
                s2 = df[col2].dropna()
                aligned = pd.concat([s1, s2], axis=1).dropna()
                if aligned.shape[0] > 0:
                    mi_score = mutual_info_score(aligned[col1], aligned[col2])
                    mi[f"{col1} vs {col2}"] = mi_score
            except Exception:
                continue
    return mi

def find_sunshower_patterns(df: pd.DataFrame) -> List[str]:
    summary = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_data = df[col].dropna()
            summary[col] = {
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'unique': col_data.nunique(),
                'missing': int(df[col].isna().sum()),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurt(),
                'outliers': int((np.abs(col_data - col_data.mean()) > 3 * col_data.std()).sum()),
                'quantiles': {str(k): v for k, v in col_data.quantile([0.25, 0.5, 0.75]).to_dict().items()},
            }
        else:
            vc = df[col].value_counts()
            probs = vc / vc.sum() if vc.sum() > 0 else vc
            entropy = float(-(probs * np.log2(probs + 1e-9)).sum())
            summary[col] = {
                'unique': df[col].nunique(),
                'top_values': {str(k): v for k, v in vc.head(5).to_dict().items()},
                'missing': int(df[col].isna().sum()),
                'entropy': entropy
            }
    # Pairwise statistics
    summary['mutual_info'] = compute_mutual_info(df)
    # Correlation matrix
    corr = df.corr(numeric_only=True).to_dict()
    serializable_summary = to_serializable(summary)
    serializable_corr = to_serializable(corr)
    # Prepare prompt for LLM
    system_prompt = (
        "You are a data scientist. Given the following summary statistics, correlations, mutual information, and advanced statistics (skewness, kurtosis, outlier counts, quantiles, entropy) for a dataset, your task is to find and describe ONLY non-obvious, surprising, counterintuitive, or quirky relationships between columns.\n"
        "DO NOT mention relationships that are expected, direct, or obvious, such as: Revenue and Profit being correlated, Expenses and Profit being negatively correlated, or any relationships that are direct mathematical consequences or commonly known in business/finance.\n"
        "Focus on patterns that a user might find unexpected, rare, or worthy of deeper exploration.\n"
        "For example, do NOT include: 'Revenue and Profit are correlated', 'Expenses and Profit are negatively correlated', or 'Tax Rate has minimal impact on Revenue.'\n"
        "Instead, include things like: 'A rarely used category has unusually high average profit', 'A specific note is associated with outlier expenses', 'A group with few entries has the highest revenue variance', or 'A category with many missing values is linked to high profits.'\n"
        "Be concise and list each insight as a bullet point. If you cannot find any non-obvious or surprising relationships, say so explicitly."
    )
    user_prompt = (
        f"Summary statistics for each column:\n{json.dumps(serializable_summary, indent=2)}\n\n"
        f"Correlation matrix:\n{json.dumps(serializable_corr, indent=2)}"
    )
    # Use OpenAI LLM
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    # Parse and return insights as a list
    content = response.choices[0].message.content
    if content is not None:
        insights = [line.strip('-• ') for line in content.split('\n') if line.strip() and (line.startswith('-') or line.startswith('•'))]
        if not insights:
            insights = [content.strip()]
    else:
        insights = []
    return insights

def to_serializable(val):
    import numpy as np
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    elif isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [to_serializable(v) for v in val]
    else:
        return val
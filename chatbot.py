# =====================================================
# HELPER FUNCTIONS
# =====================================================

def format_class_summary(class_counts):
    label_map = {
        0: "Vegetation",
        1: "Built-up",
        2: "Water",
        3: "Other"
    }

    total = sum(class_counts.values()) if class_counts else 1

    parts = []
    for k, v in class_counts.items():
        pct = (v / total) * 100 if total > 0 else 0
        parts.append(f"{label_map.get(k, k)}: {v} cells ({pct:.1f}%)")

    return ", ".join(parts)


# =====================================================
# MAIN RESPONSE FUNCTION
# =====================================================

def generate_response(user_query, context):

    user_input = user_query.lower()

    year = context.get("year")
    model = context.get("model")
    compare = context.get("compare_mode")
    accuracy = context.get("accuracy")

    class_counts = context.get("class_counts", {})
    summary = format_class_summary(class_counts)

    # =====================================================
    # PIE CHART
    # =====================================================
    if "pie" in user_input:
        return f"""
Normal Explanation:
The pie chart shows the distribution of land cover classes for {year}. The composition is: {summary}. This helps understand how much area each class occupies.

Simple Explanation:
The city is divided into slices: {summary}.
"""

    # =====================================================
    # CLASS DISTRIBUTION / BAR CHART
    # =====================================================
    if "distribution" in user_input or "bar" in user_input:
        return f"""
Normal Explanation:
The class distribution graph shows how many grid cells belong to each land cover class. The counts are: {summary}. It helps compare dominance of each class.

Simple Explanation:
It shows how many blocks are plants, buildings, or water: {summary}.
"""

    # =====================================================
    # MAP
    # =====================================================
    if "map" in user_input:
        return f"""
Normal Explanation:
The map shows spatial distribution of land cover for {year}. Each grid cell is classified into vegetation, built-up, water, or other categories.

Simple Explanation:
Each square on the map shows what type of land is there.
"""

    # =====================================================
    # ACCURACY / RELIABILITY
    # =====================================================
    if "accuracy" in user_input or "reliable" in user_input or "trust" in user_input:
        if year == "2024":
            return """
Normal Explanation:
There is no ground truth available for 2024, so model accuracy cannot be evaluated. Predictions are uncertain.

Simple Explanation:
We don’t know the real answer, so it’s just a guess.
"""
        elif accuracy is not None:
            return f"""
Normal Explanation:
The model accuracy is approximately {accuracy*100:.2f}%. While this is reasonably good, errors still exist.

Simple Explanation:
The model is mostly correct but still makes mistakes.
"""
        else:
            return """
Normal Explanation:
Accuracy information is not available in this view.

Simple Explanation:
We don’t have a score here.
"""

    # =====================================================
    # MODEL COMPARISON
    # =====================================================
    if "mlp" in user_input or "ridge" in user_input or "model" in user_input:
        return """
Normal Explanation:
MLP performs better because it captures nonlinear relationships, while Ridge is simpler and linear, making it more interpretable but less powerful.

Simple Explanation:
MLP is smarter for complex patterns, Ridge is simpler and easier to understand.
"""

    # =====================================================
    # CHANGE / DIFFERENCE
    # =====================================================
    if "change" in user_input or "difference" in user_input:
        return f"""
Normal Explanation:
Changes in land cover represent differences in class distribution over time or between models. Current distribution is: {summary}. However, unrealistic changes may occur due to dataset inconsistencies.

Simple Explanation:
If numbers change too much, it might be a data problem, not real change.
"""

    # =====================================================
    # FEATURES
    # =====================================================
    if "ndvi" in user_input or "ndbi" in user_input or "ndwi" in user_input:
        return """
Normal Explanation:
NDVI measures vegetation, NDBI indicates built-up areas, and NDWI represents water presence.

Simple Explanation:
NDVI = plants, NDBI = buildings, NDWI = water.
"""

    # =====================================================
    # GENERAL EXPLAIN
    # =====================================================
    if "explain" in user_input:
        return f"""
Normal Explanation:
This dashboard shows land cover analysis for {year}, including maps, charts, and model predictions. Current distribution is: {summary}.

Simple Explanation:
It shows what types of land exist and how much of each type is there.
"""

    # =====================================================
    # DEFAULT
    # =====================================================
    return f"""
Normal Explanation:
This dashboard shows land cover patterns and model predictions. Current distribution is: {summary}.

Simple Explanation:
Ask about maps, charts, or models.
"""


# =====================================================
# CONTEXT BUILDER
# =====================================================

def build_context(selected_year, selected_model=None, compare_mode=False, accuracy=None):

    return {
        "year": selected_year,
        "model": selected_model,
        "compare_mode": compare_mode,
        "accuracy": accuracy
    }
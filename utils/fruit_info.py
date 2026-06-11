"""
utils/fruit_info.py
Static fruit information and AI insight generation.
"""

FRUIT_DATABASE = {
    "Apple": {
        "nutrition":  "Rich in fibre, Vitamin C, and antioxidants. ~52 kcal per 100g.",
        "storage":    "Refrigerate at 0–4 °C; keeps 4–6 weeks. Keep away from ethylene-sensitive produce.",
        "handling":   "Handle gently to prevent bruising. Do not wash until ready to use.",
        "shelf_life": "3–5 weeks refrigerated · 1–2 weeks at room temp",
    },
    "Mango": {
        "nutrition":  "Excellent source of Vitamin A & C, folate, and digestive enzymes. ~60 kcal per 100g.",
        "storage":    "Store unripe at room temp. Once ripe, refrigerate for up to 5 days.",
        "handling":   "Avoid stacking; bruises easily. Check for firmness at stem end.",
        "shelf_life": "2–3 days ripe at room temp · 5 days refrigerated",
    },
    "Banana": {
        "nutrition":  "High in potassium, Vitamin B6, and fast-release carbohydrates. ~89 kcal per 100g.",
        "storage":    "Store at room temperature. Refrigerating blackens skin but preserves flesh.",
        "handling":   "Keep in bunches to slow ripening. Separate to accelerate.",
        "shelf_life": "2–5 days once yellow",
    },
    "Orange": {
        "nutrition":  "Outstanding Vitamin C source; also thiamine and folate. ~47 kcal per 100g.",
        "storage":    "Room temperature for 1 week; refrigerate up to 4 weeks.",
        "handling":   "Avoid moisture; can cause mould. Single-layer storage preferred.",
        "shelf_life": "1 week room temp · 3–4 weeks refrigerated",
    },
    "Tomato": {
        "nutrition":  "High lycopene, Vitamins C & K, and potassium. ~18 kcal per 100g.",
        "storage":    "Store at room temperature. Refrigeration dulls flavour and softens texture.",
        "handling":   "Store stem-side down to prevent moisture loss. Do not stack heavily.",
        "shelf_life": "5–7 days at room temperature",
    },
    "Strawberry": {
        "nutrition":  "Very high in Vitamin C, manganese, and antioxidants. ~32 kcal per 100g.",
        "storage":    "Refrigerate unwashed in a single layer. Use within 2–3 days.",
        "handling":   "Extremely delicate — single layer only, avoid pressure.",
        "shelf_life": "2–3 days refrigerated",
    },
    "Grape": {
        "nutrition":  "Contains resveratrol, Vitamins C & K, and natural sugars. ~69 kcal per 100g.",
        "storage":    "Keep refrigerated in original ventilated bag. Do not wash until eating.",
        "handling":   "Keep in clusters; individual removal causes faster spoilage.",
        "shelf_life": "1–2 weeks refrigerated",
    },
    "Unknown Fruit": {
        "nutrition":  "Nutritional content varies by fruit type. Consult USDA FoodData Central for details.",
        "storage":    "Generally store most fruits between 0–10 °C with proper ventilation.",
        "handling":   "Handle with clean, dry hands. Avoid bruising and exposure to direct sunlight.",
        "shelf_life": "Varies by fruit type and ripeness stage",
    },
}

AI_INSIGHTS = {
    "Good Quality": {
        "icon":  "🟢",
        "main":  "Fruit is suitable for sale, packaging, and distribution.",
        "sub":   "High confidence classification indicates uniform colour, texture, and surface integrity.",
        "tips":  [
            "Clear for immediate packing and dispatch",
            "Meets retail quality standards",
            "Prioritise for premium market channels",
            "Document batch for quality traceability records",
        ],
    },
    "Mixed Quality": {
        "icon":  "🟡",
        "main":  "Fruit shows inconsistent quality — manual inspection recommended.",
        "sub":   "Mixed signals detected; may contain a blend of good and deteriorated specimens.",
        "tips":  [
            "Sort batch manually before distribution",
            "Good specimens can be redirected to retail",
            "Marginal specimens suitable for processing/juicing",
            "Monitor storage conditions to prevent further degradation",
        ],
    },
    "Bad Quality": {
        "icon":  "🔴",
        "main":  "Fruit does not meet quality standards for direct retail distribution.",
        "sub":   "Visual indicators suggest significant defects, bruising, or early-stage spoilage.",
        "tips":  [
            "Do not dispatch to retail outlets",
            "Evaluate for industrial processing or composting",
            "Investigate storage or handling conditions",
            "Conduct root-cause analysis on this batch",
        ],
    },
}


def get_fruit_info(fruit_name: str) -> dict:
    """Return fruit info dict, falling back to Unknown Fruit."""
    return FRUIT_DATABASE.get(fruit_name, FRUIT_DATABASE["Unknown Fruit"])


def get_ai_insights(quality_label: str) -> dict:
    """Return AI insights for a given quality label."""
    return AI_INSIGHTS.get(quality_label, AI_INSIGHTS["Mixed Quality"])

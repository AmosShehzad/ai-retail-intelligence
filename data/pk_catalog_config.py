"""
Master dictionary for Pakistani retail product generation.
All prices in PKR. All values based on actual market data.
"""

CATALOG = {
    "Tea & Beverages": {
        "brands"    : ["Tapal", "Lipton", "Tetley", "Supreme", "Vital"],
        "products"  : [
            {"name": "Danedar",         "variants": ["95g", "200g", "500g"],     "base_price": 100},
            {"name": "Family Mixture",  "variants": ["95g", "200g"],             "base_price": 90},
            {"name": "Green Tea",       "variants": ["30 bags", "100 bags"],     "base_price": 130},
            {"name": "Elaichi Tea",     "variants": ["95g", "190g"],             "base_price": 115},
        ],
        "cost_margin": 0.72,  # cost = selling_price * this factor
    },
    "Spices & Masala": {
        "brands"    : ["Shan", "National", "Ahmed", "Mehran", "Laziza"],
        "products"  : [
            {"name": "Biryani Masala",  "variants": ["50g", "100g", "200g"],     "base_price": 75},
            {"name": "Karahi Masala",   "variants": ["50g", "100g"],             "base_price": 70},
            {"name": "Nihari Masala",   "variants": ["50g", "100g"],             "base_price": 65},
            {"name": "Chaat Masala",    "variants": ["50g", "100g"],             "base_price": 60},
            {"name": "Tikka Masala",    "variants": ["50g", "100g"],             "base_price": 72},
        ],
        "cost_margin": 0.70,
    },
    "Dairy": {
        "brands"    : ["Nestlé MilkPak", "Olpers", "Good Milk", "Haleeb", "Day Fresh"],
        "products"  : [
            {"name": "Full Cream Milk", "variants": ["250ml", "500ml", "1L"],    "base_price": 60},
            {"name": "Skimmed Milk",    "variants": ["500ml", "1L"],             "base_price": 55},
            {"name": "Cream",           "variants": ["200ml"],                   "base_price": 120},
            {"name": "Butter",          "variants": ["100g", "200g"],            "base_price": 180},
        ],
        "cost_margin": 0.80,
    },
    "Cooking Oil": {
        "brands"    : ["Dalda", "Sufi", "Habib", "Seasons", "Tullo"],
        "products"  : [
            {"name": "Cooking Oil",     "variants": ["1L", "2L", "5L"],         "base_price": 400},
            {"name": "Banaspati Ghee",  "variants": ["1kg", "2.5kg"],           "base_price": 450},
            {"name": "Canola Oil",      "variants": ["1L", "3L"],               "base_price": 480},
        ],
        "cost_margin": 0.82,
    },
    "Detergents": {
        "brands"    : ["Surf Excel", "Ariel", "Bonus", "Brite", "Express"],
        "products"  : [
            {"name": "Washing Powder",  "variants": ["500g", "1kg", "2kg"],     "base_price": 300},
            {"name": "Liquid Detergent","variants": ["500ml", "1L"],            "base_price": 350},
            {"name": "Bar Soap",        "variants": ["150g", "250g"],           "base_price": 80},
        ],
        "cost_margin": 0.73,
    },
    "Personal Care": {
        "brands"    : ["Lux", "Safeguard", "Dettol", "Lifebuoy", "Dove"],
        "products"  : [
            {"name": "Soap Bar",        "variants": ["100g", "150g", "175g"],   "base_price": 90},
            {"name": "Hand Wash",       "variants": ["200ml", "500ml"],         "base_price": 180},
            {"name": "Shampoo",         "variants": ["180ml", "360ml"],         "base_price": 250},
            {"name": "Body Lotion",     "variants": ["200ml", "400ml"],         "base_price": 300},
        ],
        "cost_margin": 0.70,
    },
    "Instant Food": {
        "brands"    : ["Knorr", "Shan", "National", "Kolson", "Indomie"],
        "products"  : [
            {"name": "Chicken Noodles", "variants": ["60g", "pack of 6"],       "base_price": 55},
            {"name": "Masala Noodles",  "variants": ["60g", "pack of 6"],       "base_price": 55},
            {"name": "Soup Mix",        "variants": ["60g", "125g"],            "base_price": 70},
            {"name": "Tomato Ketchup",  "variants": ["300g", "500g", "800g"],   "base_price": 120},
        ],
        "cost_margin": 0.74,
    },
    "Beverages": {
        "brands"    : ["Rooh Afza", "Tang", "Nestle Milo", "Pakola", "Shezan"],
        "products"  : [
            {"name": "Squash",          "variants": ["500ml", "800ml"],         "base_price": 280},
            {"name": "Drink Mix",       "variants": ["125g", "500g"],           "base_price": 160},
            {"name": "Fruit Juice",     "variants": ["200ml", "1L"],            "base_price": 90},
        ],
        "cost_margin": 0.73,
    },
    "Snacks": {
        "brands"    : ["Kolson", "Lay's", "Peek Freans", "EBM", "Bisconni"],
        "products"  : [
            {"name": "Biscuits",        "variants": ["112g", "230g"],           "base_price": 70},
            {"name": "Chips",           "variants": ["30g", "80g"],             "base_price": 50},
            {"name": "Crackers",        "variants": ["100g", "200g"],           "base_price": 80},
        ],
        "cost_margin": 0.71,
    },
    "Condiments": {
        "brands"    : ["National", "Shan", "Ahmed", "Mitchells", "Ravi"],
        "products"  : [
            {"name": "Mango Pickle",    "variants": ["400g", "800g"],           "base_price": 160},
            {"name": "Mixed Pickle",    "variants": ["400g"],                   "base_price": 140},
            {"name": "Chilli Sauce",    "variants": ["300g", "500g"],           "base_price": 120},
            {"name": "Tamarind Paste",  "variants": ["200g"],                   "base_price": 100},
        ],
        "cost_margin": 0.73,
    },
}

# Stock range per category (min, max units)
STOCK_RANGES = {
    "Tea & Beverages" : (20, 120),
    "Spices & Masala" : (30, 150),
    "Dairy"           : (10, 60),
    "Cooking Oil"     : (15, 80),
    "Detergents"      : (20, 100),
    "Personal Care"   : (25, 120),
    "Instant Food"    : (40, 200),
    "Beverages"       : (15, 90),
    "Snacks"          : (30, 180),
    "Condiments"      : (20, 100),
}

# Variant size → price multiplier (bigger pack = higher price, not linear)
SIZE_MULTIPLIERS = {
    "95g"       : 1.0,
    "100g"      : 1.0,
    "112g"      : 1.1,
    "125g"      : 1.2,
    "150g"      : 1.4,
    "175g"      : 1.5,
    "190g"      : 1.8,
    "200g"      : 1.9,
    "200ml"     : 1.0,
    "230g"      : 2.1,
    "250g"      : 2.2,
    "250ml"     : 1.1,
    "300g"      : 2.5,
    "360ml"     : 1.8,
    "400g"      : 3.2,
    "400ml"     : 1.9,
    "500g"      : 4.0,
    "500ml"     : 2.5,
    "800g"      : 5.5,
    "800ml"     : 4.0,
    "1L"        : 4.5,
    "1kg"       : 4.5,
    "2L"        : 7.5,
    "2.5kg"     : 9.0,
    "2kg"       : 8.0,
    "3L"        : 10.0,
    "5L"        : 16.0,
    "30 bags"   : 1.3,
    "60g"       : 0.6,
    "30g"       : 0.5,
    "80g"       : 0.8,
    "pack of 6" : 5.0,
    "100 bags"  : 3.5,
}
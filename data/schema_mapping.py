# Maps Kaggle dataset columns → your project's standard schema
# Change only this file if you ever swap datasets

KAGGLE_TO_STANDARD = {
    "StockCode"   : "product_code",
    "Description" : "product_name",
    "Quantity"    : "quantity",
    "Price"       : "unit_price_gbp",
    "UnitPrice"   : "unit_price_gbp",
    "InvoiceDate" : "sale_date",
    "Invoice"     : "invoice_id",
    "Country"     : "country",
    "Customer ID" : "customer_id",
}

# PKR conversion rate (GBP → PKR, approximate)
GBP_TO_PKR = 350.0

# Filters applied during cleaning
FILTERS = {
    "min_quantity"   : 1,       # drop cancelled orders (negative qty)
    "min_unit_price" : 0.01,    # drop zero-price test entries
    "country"        : None,    # None = keep all countries
}

# Output column order for processed CSV
OUTPUT_COLUMNS = [
    "product_code",
    "product_name",
    "category",          # derived via categorizer
    "quantity",
    "unit_price_pkr",
    "sale_date",
    "invoice_id",
]
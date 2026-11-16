SEX = "SEX"
EDUCATION = "EDUCATION"
MARRIAGE = "MARRIAGE"
AGE = "AGE"
LIMIT = "LIMIT_BAL"
TARGET = "default"

PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]

RAW_FEATURES = [LIMIT, SEX, EDUCATION, MARRIAGE, AGE, *PAY_STATUS, *BILL_AMT, *PAY_AMT]

# Allowed values after cleaning; used by validation and by the API schema.
CATEGORY_VALUES = {
    SEX: {1: "male", 2: "female"},
    EDUCATION: {1: "graduate school", 2: "university", 3: "high school", 4: "other"},
    MARRIAGE: {1: "married", 2: "single", 3: "other"},
}
PAY_STATUS_RANGE = (-2, 9)
AGE_RANGE = (18, 100)

from pydantic import BaseModel, validator
from typing import List

class TestCase(BaseModel):
    question:     str
    ground_truth: str

    @validator("question", "ground_truth")
    @classmethod
    def must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty or whitespace")
        return v.strip()

    @validator("question")
    @classmethod
    def must_be_a_question(cls, v):
        if not v.endswith("?"):
            raise ValueError(f"Question must end with '?': '{v}'")
        return v

class EvaluationDataset(BaseModel):
    test_cases: List[TestCase]

    @validator("test_cases")
    @classmethod
    def must_have_cases(cls, v):
        if len(v) == 0:
            raise ValueError("Dataset must contain at least one test case")
        return v

    def summary(self):
        print(f"Total test cases: {len(self.test_cases)}")
        for i, case in enumerate(self.test_cases, 1):
            print(f"{i}. Q: {case.question}")

RAW_TEST_CASES = [
    {"question": "How many days do customers have to return a product?", "ground_truth": "Customers have 30 days from the date of purchase to return products."},
    {"question": "What is required to make a return?", "ground_truth": "Customers must provide proof of purchase for all returns."},
    {"question": "How long does a refund take to process?", "ground_truth": "Refunds are processed within 5-7 business days after approval."},
    {"question": "How long does standard shipping take?", "ground_truth": "Standard shipping takes 3-5 business days."},
    {"question": "How much does express shipping cost?", "ground_truth": "Express shipping costs 9.99 pounds and takes 1-2 business days."},
    {"question": "What is the minimum order for free shipping?", "ground_truth": "Free shipping is available on orders over 50 pounds."},
    {"question": "What are the customer support hours?", "ground_truth": "Customer support is available Monday to Friday 9am to 5pm."},
    {"question": "What is the customer support email address?", "ground_truth": "The customer support email address is support@company.com."},
]

def get_test_data() -> EvaluationDataset:
    return EvaluationDataset(
        test_cases=[TestCase(**case) for case in RAW_TEST_CASES]
    )

if __name__ == "__main__":
    dataset = get_test_data()
    dataset.summary()

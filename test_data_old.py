# # ── TEST DATA ───────────────────────────────────────────
# Ground truth questions and answers based on company_policy.pdf
# Ground truth = what the CORRECT answer should be
# RAGAS compares RAG answers against these to score quality

# docstring document the function
# What the function returns
# Structure of the data
def get_test_data():
    """
    Returns list of test cases.
    Each test case has :
    question     -> what we ask the RAG system
    ground_truth -> what the correct answer should be"""
    test_cases = [
        {
            "question": "How many days do customers have to return a product?",
            "ground_truth": "Customers have 30 days from the date of purchase to return products."
        },
        {
            "question": "what is required to make a return?",
            "ground_truth": "Customers must provide proof of purchase for all returns."
        },
        {
            "question": "How long does a refund take to process?",
            "ground_truth": "Refunds are processed withing 5-7 buiness days after approval."
        },
        {
            "question": " how long does standard shipping take?",
            "ground_truth": "Standard shipping takes 3-5 business days."
        },
        {
            "question": "how long does express shipping cost?",
            "ground_truth": "Express shipping costs 9.99 pounds and takes 1-2 buiness days."
        },
        {
            "question": "what are the customer rupport hours?",
            "ground_truth": "Customer support is available Monday to Friday 9am to 5pm."
        },
        {
            "question": "what is the customer support email address?",
            "ground_truth": "The customer support email address is support @company.com."
        }
    ]
    return test_cases

if __name__ == "__main__":
    data = get_test_data()
    print(f"Totdal test cases: {len(data)}")
    for i, case in enumerate(data, 1):
        print(f"\n{i}. Q: {case['question']}")
        print(f"   A: {case['ground_truth']}")
    
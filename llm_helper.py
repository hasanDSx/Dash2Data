import json
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError

MODELS = ["gemini-3.6-flash"]
RETRYABLE_CODES = (429, 500, 503, 504)
MAX_RETRIES = 5


def _generate_with_retry(client, contents, config):
    last_error = None
    for model in MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except APIError as e:
                if getattr(e, "code", None) in RETRYABLE_CODES:
                    last_error = e
                    time.sleep(2 ** attempt)
                    continue
                raise
    raise last_error


def _normalize_records(data):
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return value
        return [data]
    return data


def extract_dashboard_data(
    images_data: list,
    api_key: str,
    num_rows: int = 50,
    date_format: str = "YYYY-MM-DD",
    include_nulls: bool = False,
) -> list:
    client = genai.Client(api_key=api_key.strip())

    if include_nulls:
        null_rule = "Introduce occasional missing values (about 3-5%) in non-key columns only."
    else:
        null_rule = "Do not include any null or missing values."

    prompt = f"""
    You are an expert data analyst and AI vision specialist.
    Analyze the provided dashboard image(s) thoroughly:
    1. Infer all required underlying columns necessary to recreate the dashboard visuals across all provided images (e.g., Date, Region, Product, Sales, Profit, KPIs, Percentages, etc.).
    2. Understand the mathematical relationships, card aggregates (KPIs), and percentage metrics displayed across all screenshots.
    3. Generate realistic mock data in a single consolidated tabular format containing approximately {num_rows} rows.
    4. Mandatory condition: When performing aggregations (Group By, Sum, Average) on this dataset, the calculated metrics must closely match the numbers and figures shown in the provided dashboard images.
    5. Format all date columns as {date_format}.
    6. {null_rule}

    Return the result strictly as a valid JSON array of flat objects, where every object has the same keys, without any explanations or additional text:
    [
      {{"Column1": "Value1", "Column2": "Value2"}},
      ...
    ]
    """

    contents = []
    for img_bytes, mime_type in images_data:
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
    contents.append(prompt)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.3,
    )

    response = _generate_with_retry(client, contents, config)
    return _normalize_records(json.loads(response.text))

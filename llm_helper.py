import json
from google import genai
from google.genai import types

def extract_dashboard_data(images_data: list, api_key: str, num_rows: int = 50) -> list:
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an expert data analyst and AI vision specialist.
    Analyze the provided dashboard image(s) thoroughly:
    1. Infer all required underlying columns necessary to recreate the dashboard visuals across all provided images (e.g., Date, Region, Product, Sales, Profit, KPIs, Percentages, etc.).
    2. Understand the mathematical relationships, card aggregates (KPIs), and percentage metrics displayed across all screenshots.
    3. Generate realistic mock data in a single consolidated tabular format containing approximately {num_rows} rows.
    4. Mandatory condition: When performing aggregations (Group By, Sum, Average) on this dataset, the calculated metrics must closely match the numbers and figures shown in the provided dashboard images.

    Return the result strictly as a valid JSON list of objects, without any markdown formatting, explanations, or additional text:
    [
      {{"Column1": "Value1", "Column2": "Value2"}},
      ...
    ]
    """

    contents = []
    for img_bytes, mime_type in images_data:
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
    contents.append(prompt)

    # Updated model string to gemini-3.6-flash
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=contents
    )

    response_text = response.text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]

    return json.loads(response_text.strip())
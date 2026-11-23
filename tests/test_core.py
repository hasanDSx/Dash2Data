import io
import json
import math
import os
import unittest
import zipfile

import pandas as pd

from core.calibrator import derive_column_targets
from core.dax import build_dax, q_col, q_str, to_dax_text
from core.exporter import build_excel, build_zip
from core.extractor import parse_json_text
from core.measures import canon_value, make_evaluator
from core.model import build_model
from core.numbers import parse_number
from core.pipeline import build_dataset
from core.schemas import SpecError, load_spec

CASES = os.path.join(os.path.dirname(__file__), "..", "benchmark", "cases")


def load_case(name):
    with open(os.path.join(CASES, f"{name}.spec.json"), encoding="utf-8") as f:
        return json.load(f)


def small_spec(**over):
    d = {
        "title": "t",
        "date": {"name": "Date", "start": "2024-01-01", "end": "2024-06-30"},
        "dimensions": [{"name": "Region", "values": ["A", "B", "C"]}],
        "fact_columns": [{"name": "Sales"}, {"name": "Cost", "derived_from": "Sales", "derived_ratio": 0.6}],
        "measures": [
            {"name": "Total Sales", "kind": "agg", "agg": "SUM", "column": "Sales", "format": "currency"},
            {"name": "Total Cost", "kind": "agg", "agg": "SUM", "column": "Cost"},
            {"name": "Profit", "kind": "diff", "left": "Total Sales", "right": "Total Cost"},
            {"name": "Margin", "kind": "ratio", "num": "Profit", "den": "Total Sales", "format": "percent"},
        ],
        "targets": [
            {"measure": "Total Sales", "value": 100000},
            {"measure": "Margin", "value": 0.25},
        ],
    }
    d.update(over)
    return d


class TestNumbers(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_number("1.2M"), 1_200_000)
        self.assertEqual(parse_number("$3,400"), 3400)
        self.assertAlmostEqual(parse_number("23%"), 0.23)
        self.assertEqual(parse_number("(500)"), -500)
        self.assertEqual(parse_number(7), 7.0)
        self.assertEqual(parse_number("2.5k"), 2500)

    def test_invalid(self):
        for bad in ("abc", "", None, True, float("nan")):
            with self.assertRaises(ValueError):
                parse_number(bad)


class TestSchemas(unittest.TestCase):
    def test_unknown_numeric_column_is_added(self):
        d = small_spec()
        d["measures"].append({"name": "Qty", "kind": "agg", "agg": "SUM", "column": "Quantity"})
        spec, w = load_spec(d)
        self.assertIn("Quantity", spec.fact_map())
        self.assertTrue(any("Quantity" in x for x in w))

    def test_circular_measures_dropped(self):
        d = small_spec()
        d["measures"] += [
            {"name": "X", "kind": "ratio", "num": "Y", "den": "Total Sales"},
            {"name": "Y", "kind": "ratio", "num": "X", "den": "Total Sales"},
        ]
        spec, w = load_spec(d)
        self.assertNotIn("X", spec.measure_map())
        self.assertNotIn("Y", spec.measure_map())

    def test_percent_heuristic(self):
        d = small_spec()
        d["targets"][1]["value"] = 25
        spec, _ = load_spec(d)
        self.assertAlmostEqual([t for t in spec.targets if t.measure == "Margin"][0].value, 0.25)

    def test_reserved_dimension_removed(self):
        d = small_spec()
        d["dimensions"].append({"name": "Year", "values": ["2024"]})
        spec, w = load_spec(d)
        self.assertNotIn("Year", spec.dim_map())

    def test_no_targets_raises(self):
        with self.assertRaises(SpecError):
            load_spec(small_spec(targets=[]))

    def test_bad_target_value_dropped(self):
        d = small_spec()
        d["targets"].append({"measure": "Total Sales", "value": "n/a", "filters": [{"column": "Region", "value": "A"}]})
        spec, w = load_spec(d)
        self.assertEqual(len(spec.targets), 2)

    def test_unseen_dimension_member_added(self):
        d = small_spec()
        d["targets"].append({"measure": "Total Sales", "value": 10, "filters": [{"column": "Region", "value": "Z"}]})
        spec, _ = load_spec(d)
        self.assertIn("Z", spec.dim_map()["Region"].values)

    def test_filters_as_dict_accepted(self):
        d = small_spec()
        d["targets"].append({"measure": "Total Sales", "value": 10, "filters": {"Region": "A"}})
        spec, _ = load_spec(d)
        self.assertEqual(spec.targets[-1].filters[0].column, "Region")


class TestEvaluator(unittest.TestCase):
    def test_canon(self):
        self.assertEqual(canon_value("Month", "January"), "jan")
        self.assertEqual(canon_value("Month", "3"), "mar")
        self.assertEqual(canon_value("Quarter", "Quarter 2"), "q2")
        self.assertEqual(canon_value("Year", "2023.0"), "2023")
        self.assertEqual(canon_value("YearMonth", "Jan 2023"), "2023-01")


class TestTargetResolution(unittest.TestCase):
    def test_margin_resolves_cost(self):
        spec, _ = load_spec(small_spec())
        ct, _ = derive_column_targets(spec)
        cost = [t for t in ct if t.column == "Cost"][0]
        self.assertAlmostEqual(cost.value, 75000.0)

    def test_group_ratio_resolves_per_group(self):
        d = small_spec()
        d["targets"] += [
            {"measure": "Total Sales", "value": 40000, "filters": [{"column": "Region", "value": "A"}]},
            {"measure": "Margin", "value": 0.5, "filters": [{"column": "Region", "value": "A"}]},
        ]
        spec, _ = load_spec(d)
        ct, _ = derive_column_targets(spec)
        cost_a = [t for t in ct if t.column == "Cost" and t.filters][0]
        self.assertAlmostEqual(cost_a.value, 20000.0)


class TestPipeline(unittest.TestCase):
    def test_retail_all_targets_match(self):
        spec, _ = load_spec(load_case("retail"))
        r = build_dataset(spec, num_rows=50, seed=7)
        self.assertEqual(r.summary["off"], 0)
        self.assertLess(r.summary["max_error_pct"], 0.01)
        self.assertEqual(r.n_rows, 1500)           # auto-sized from Orders KPI

    def test_fixed_rows_flags_unreachable_counts(self):
        spec, _ = load_spec(load_case("retail"))
        r = build_dataset(spec, num_rows=100, auto_rows=False, seed=7)
        self.assertEqual(r.n_rows, 100)
        self.assertTrue(any("cannot match" in w for w in r.warnings))
        orders = r.validation[r.validation.Measure == "Orders"].iloc[0]
        self.assertNotEqual(orders["Status"], "✅ match")

    def test_deterministic(self):
        spec, _ = load_spec(small_spec())
        a = build_dataset(spec, num_rows=80, seed=3).flat
        b = build_dataset(spec, num_rows=80, seed=3).flat
        pd.testing.assert_frame_equal(a, b)

    def test_nulls_still_match(self):
        spec, _ = load_spec(small_spec())
        r = build_dataset(spec, num_rows=200, include_nulls=True, seed=5)
        self.assertGreater(r.flat["Sales"].isna().sum(), 0)
        self.assertLess(r.summary["max_error_pct"], 0.05)

    def test_no_negative_values_and_integer_quantity(self):
        spec, _ = load_spec(load_case("retail"))
        r = build_dataset(spec, seed=11)
        self.assertTrue((r.flat["Sales"] >= 0).all())
        self.assertTrue(str(r.flat["Quantity"].dtype).lower().startswith("int"))

    def test_inconsistent_targets_compromise(self):
        d = small_spec()
        d["targets"] = [{"measure": "Total Sales", "value": 100000}] + [
            {"measure": "Total Sales", "value": 40000, "filters": [{"column": "Region", "value": r}]}
            for r in "ABC"]                                  # groups sum to 120k != 100k
        spec, _ = load_spec(d)
        r = build_dataset(spec, num_rows=300, seed=2)
        self.assertGreater(r.summary["max_error_pct"], 1.0)  # honest report, no crash

    def test_partial_chart_gets_residual(self):
        d = small_spec()
        d["targets"] = [{"measure": "Total Sales", "value": 100000},
                        {"measure": "Total Sales", "value": 50000, "filters": [{"column": "Region", "value": "A"}]}]
        spec, _ = load_spec(d)
        r = build_dataset(spec, num_rows=300, seed=2)
        self.assertEqual(r.summary["off"], 0)
        rest = r.flat[r.flat.Region != "A"]["Sales"].sum()
        self.assertAlmostEqual(rest, 50000, delta=1)

    def test_other_cases(self):
        for name in ("saas", "hr"):
            spec, _ = load_spec(load_case(name))
            r = build_dataset(spec, num_rows=200, seed=1)
            self.assertEqual(r.summary["off"], 0, name)


class TestModelAndDax(unittest.TestCase):
    def setUp(self):
        self.spec, _ = load_spec(load_case("retail"))
        self.res = build_dataset(self.spec, seed=1)

    def test_star_schema_integrity(self):
        m = build_model(self.res.flat, self.spec, "star")
        fact = m.tables["Fact_Sales"]
        for rel in m.relationships:
            if rel["to_table"] == "Dim_Date":
                self.assertTrue(fact["Order Date"].isin(m.tables["Dim_Date"]["Order Date"]).all())
            else:
                dim = m.tables[rel["to_table"]]
                self.assertTrue(fact[rel["from_column"]].isin(dim[rel["to_column"]]).all())
                self.assertTrue(dim[rel["to_column"]].is_unique)
        self.assertNotIn("Region", fact.columns)
        self.assertNotIn("Month", fact.columns)

    def test_star_reproduces_flat_totals(self):
        m = build_model(self.res.flat, self.spec, "star")
        fact, dim = m.tables["Fact_Sales"], m.tables["Dim_Region"]
        joined = fact.merge(dim, on="RegionKey")
        north = joined[joined.Region == "North"]["Sales"].sum()
        self.assertAlmostEqual(north, 400000, delta=1)

    def test_flat_has_no_date_parts_or_derived_columns(self):
        m = build_model(self.res.flat, self.spec, "flat")
        cols = set(m.tables["Fact_Sales"].columns)
        self.assertFalse(cols & {"Year", "Month", "Quarter", "Profit", "Margin"})

    def test_dax_star_and_flat(self):
        star = build_model(self.res.flat, self.spec, "star")
        d = {x.name: x for x in build_dax(self.spec, star, self.res.validation)}
        self.assertIn("'Dim_Product'[Category] = \"Technology\"", d["Tech Sales"].expression)
        self.assertEqual(d["Profit Margin"].expression, "DIVIDE ( [Total Profit], [Total Sales] )")
        self.assertTrue(d["Total Sales"].validated)
        self.assertIn("TOTALYTD ( [Total Sales], 'Dim_Date'[Order Date] )", d["Total Sales YTD"].expression)
        flat = build_model(self.res.flat, self.spec, "flat")
        d2 = {x.name: x for x in build_dax(self.spec, flat, self.res.validation)}
        self.assertIn("'Fact_Sales'[Category]", d2["Tech Sales"].expression)

    def test_flat_date_part_filter(self):
        spec_d = small_spec()
        spec_d["measures"].append({"name": "Jan Sales", "kind": "agg", "agg": "SUM", "column": "Sales",
                                   "filters": [{"column": "Month", "value": "January"}]})
        spec, _ = load_spec(spec_d)
        r = build_dataset(spec, num_rows=100, seed=1)
        flat = build_model(r.flat, spec, "flat")
        d = {x.name: x for x in build_dax(spec, flat, r.validation, recommended=False)}
        self.assertIn('FORMAT ( \'Fact_Sales\'[Date], "mmm" ) = "Jan"', d["Jan Sales"].expression)

    def test_recommended_toggle(self):
        m = build_model(self.res.flat, self.spec, "star")
        only = build_dax(self.spec, m, self.res.validation, recommended=False)
        full = build_dax(self.spec, m, self.res.validation, recommended=True)
        self.assertTrue(all(x.origin == "dashboard" for x in only))
        self.assertGreater(len(full), len(only))
        self.assertTrue(all(not x.validated for x in full if x.origin == "recommended"))

    def test_quoting(self):
        self.assertEqual(q_col("T'x", "a]b"), "'T''x'[a]]b]")
        self.assertEqual(q_str('say "hi"'), '"say ""hi"""')

    def test_measure_name_colliding_with_column_is_renamed(self):
        d = small_spec()
        d["measures"][0]["name"] = "Sales"           # same as the column
        d["targets"][0]["measure"] = "Sales"
        spec, _ = load_spec(d)
        r = build_dataset(spec, num_rows=50, seed=1)
        m = build_model(r.flat, spec, "flat")
        names = [x.name for x in build_dax(spec, m, r.validation, recommended=False)]
        self.assertNotIn("Sales", names)
        self.assertIn("Total Sales", names)


class TestExport(unittest.TestCase):
    def test_zip_and_excel(self):
        spec, _ = load_spec(load_case("retail"))
        r = build_dataset(spec, seed=1)
        m = build_model(r.flat, spec, "star")
        d = build_dax(spec, m, r.validation)
        z = zipfile.ZipFile(io.BytesIO(build_zip(m, d, r.validation, r.summary, spec, "DD/MM/YYYY", "my out")))
        names = z.namelist()
        self.assertIn("my_out/measures.dax", names)
        self.assertIn("my_out/Fact_Sales.csv", names)
        csv = pd.read_csv(io.BytesIO(z.read("my_out/Fact_Sales.csv")))
        self.assertRegex(str(csv["Order Date"].iloc[0]), r"^\d{2}/\d{2}/\d{4}$")
        xl = build_excel(m, d, r.validation, "YYYY-MM-DD")
        self.assertGreater(len(xl), 5000)
        self.assertIn("Total Sales =", to_dax_text(d, m))


class TestExtractorHelpers(unittest.TestCase):
    def test_parse_json_text(self):
        self.assertEqual(parse_json_text('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_text('[{"a": 1}]'), {"a": 1})
        with self.assertRaises(Exception):
            parse_json_text("not json")


if __name__ == "__main__":
    unittest.main()

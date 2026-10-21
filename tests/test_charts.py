import unittest
from unittest.mock import patch

import pandas as pd

from src.charts import (
    bar_figure,
    confusion_matrix_figure,
    glass_figure,
    line_figure,
    model_metric_figure,
)


class GlassChartTests(unittest.TestCase):
    def test_chart_base_supports_dark_and_light_appearance(self):
        dark = glass_figure(mode="Dark")
        light = glass_figure(mode="Light")
        self.assertEqual(dark.layout.font.color, "#e9edfa")
        self.assertEqual(light.layout.font.color, "#202842")
        self.assertEqual(light.layout.plot_bgcolor, "rgba(255,255,255,.08)")

    def test_chart_base_defaults_to_dark_before_theme_widget_has_a_value(self):
        with patch("src.charts.get_theme_mode", return_value=None):
            figure = glass_figure()
        self.assertEqual(figure.layout.font.color, "#e9edfa")

    def test_score_bars_keep_real_values_on_transparent_canvas(self):
        # Pin the palette so this invariant stays independent of the user's
        # configured default appearance (currently Light).
        with patch("src.charts.get_theme_mode", return_value="Dark"):
            figure = bar_figure(["LI 1", "LI 2"], [0.7, 0.3], name="Score")
        self.assertEqual(list(figure.data[0].y), [0.7, 0.3])
        self.assertEqual(figure.layout.paper_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(figure.layout.plot_bgcolor, "rgba(255,255,255,0.012)")

    def test_line_chart_preserves_values_and_threshold(self):
        frame = pd.DataFrame({"time_seconds": [0.0, 1.0], "angle": [20.0, 55.0]})
        figure = line_figure(
            frame,
            "time_seconds",
            ["angle"],
            threshold=45,
            threshold_label="Review threshold",
        )
        self.assertEqual(list(figure.data[0].y), [20.0, 55.0])
        self.assertEqual(figure.layout.shapes[0].y0, 45)

    def test_model_comparison_and_confusion_matrix_have_readable_axes(self):
        comparison = model_metric_figure(
            ["Logistic Regression", "Random Forest"], [76.0, 80.0], "Accuracy", percent=True
        )
        matrix = confusion_matrix_figure([[8, 2], [1, 9]], ["LI 1", "LI 2"])
        self.assertEqual(list(comparison.data[0].y), [76.0, 80.0])
        self.assertEqual(list(matrix.data[0].z[0]), [8, 2])
        self.assertEqual(matrix.layout.xaxis.title.text, "Predicted condition")
        self.assertEqual(matrix.layout.yaxis.title.text, "Actual condition")


if __name__ == "__main__":
    unittest.main()

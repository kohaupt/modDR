from dataclasses import dataclass
from typing import Any

import plotly.express as px  # type: ignore
from dash import Dash, Input, Output, callback, dcc, html  # type: ignore

from embedding_obj import EmbeddingObj  # type: ignore


@dataclass
class DashOverlay:
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    data: list[EmbeddingObj]
    iterations: list[int]
    color_range: list[float]

    def __init__(self, data: list[EmbeddingObj]):
        self.data = data

        self.iterations = []
        for d in data:
            self.iterations.append(d.obj_id)

        self.color_range = self.compute_color_scale()
        self.instanciate_dash()

    def instanciate_dash(self) -> None:
        self.app = Dash(__name__, external_stylesheets=self.external_stylesheets)
        self.app.layout = html.Div(
            [
                html.Div(
                    [dcc.Graph(id="crossfilter-indicator-scatter")],
                    style={
                        "width": "100%",
                        "display": "inline-block",
                        "padding": "0 20",
                    },
                ),
                html.Div(
                    dcc.Slider(
                        self.iterations[0],
                        self.iterations[-1],
                        step=None,
                        value=self.iterations[0],
                        marks={
                            str(iteration): str(iteration)
                            for iteration in self.iterations
                        },
                        id="crossfilter-iteration--slider",
                    ),
                ),
            ]
        )

        @callback(
            Output("crossfilter-indicator-scatter", "figure"),
            Input("crossfilter-iteration--slider", "value"),
        )  # type: ignore
        def update_graph(iteration_value: int) -> px.scatter:
            fig = px.scatter(
                x=self.data[self.iterations.index(iteration_value)].embedding[:, 0],
                y=self.data[self.iterations.index(iteration_value)].embedding[:, 1],
                color=self.data[self.iterations.index(iteration_value)].m_jaccard,
                color_continuous_scale="inferno",
                range_color=self.color_range,
                labels={"color": "Metrics Score"},
            )

            fig.update_layout(
                margin={"l": 40, "b": 40, "t": 10, "r": 0}, hovermode="closest"
            )

            return fig

    def compute_color_scale(self) -> list[float]:
        scores = []
        for i in range(len(self.data)):
            scores.extend(self.data[i].m_jaccard)

        return [min(scores), max(scores)]

    def run(self, **kwargs: dict[str, Any]) -> None:
        self.app.run(**kwargs)
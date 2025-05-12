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
                    [
                        dcc.Graph(
                            id="crossfilter-indicator-scatter",
                            style={"width": "100%", "height": "500px"},
                        )
                    ],
                    style={
                        "width": "100%",
                        "display": "inline-block",
                        "padding": "0 20",
                    },
                ),
                html.Div(
                    dcc.Slider(
                        min=min(self.iterations),
                        max=max(self.iterations),
                        step=1,
                        value=min(self.iterations),
                        marks={
                            str(iteration): str(iteration)
                            for iteration in self.iterations
                        },
                        id="crossfilter-iteration--slider",
                    ),
                ),
            ],
            style={
                "width": "90%",
            },
        )

        @callback(
            Output("crossfilter-indicator-scatter", "figure"),
            Input("crossfilter-iteration--slider", "value"),
        )  # type: ignore
        def update_graph(iteration_value: int) -> px.scatter:
            fig = px.scatter(
                x=[
                    coord[0]
                    for coord in list(
                        [x for x in self.data if x.obj_id == iteration_value][
                            0
                        ].embedding.values()
                    )
                ],
                y=[
                    coord[1]
                    for coord in list(
                        [x for x in self.data if x.obj_id == iteration_value][
                            0
                        ].embedding.values()
                    )
                ],
                # color=self.data[self.iterations.index(iteration_value)].labels,
                # color_continuous_scale="inferno",
                # range_color=self.color_range,
                # labels={"color": "Metrics Score"},
                opacity=0.8,
            )

            fig.update_layout(
                margin={"l": 40, "b": 40, "t": 10, "r": 0},
                hovermode="closest",
                plot_bgcolor="white",
            )

            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            return fig

    def compute_color_scale(self) -> list[float]:
        scores = []
        for i in range(len(self.data)):
            scores.extend(list(self.data[i].labels.values()))

        return [min(scores), max(scores)]

    def run(self, **kwargs: dict[str, Any]) -> None:
        self.app.run(**kwargs)

from dataclasses import dataclass

import plotly.express as px
from dash import Dash, Input, Output, callback, dcc, html


@dataclass
class DashOverlay:
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    data: list
    iterations: list
    color_range: list

    def __init__(self, data, iterations):
        self.data = data
        self.iterations = iterations
        self.color_range = self.compute_color_scale()
        self.instanciate_dash()

    def instanciate_dash(self):
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
        )
        def update_graph(iteration_value):
            fig = px.scatter(
                x=self.data[self.iterations.index(iteration_value)].loc[:, "x"],
                y=self.data[self.iterations.index(iteration_value)].loc[:, "y"],
                color=self.data[self.iterations.index(iteration_value)][
                    "metrics_score"
                ],
                color_continuous_scale="inferno",
                range_color=self.color_range,
                labels={"color": "Metrics Score"},
            )

            fig.update_layout(
                margin={"l": 40, "b": 40, "t": 10, "r": 0}, hovermode="closest"
            )

            return fig

    def compute_color_scale(self):
        scores = []
        for i in range(len(self.data)):
            scores.extend(self.data[i]["metrics_score"].values)

        return [min(scores), max(scores)]

    def run(self, **kwargs):
        self.app.run(**kwargs)

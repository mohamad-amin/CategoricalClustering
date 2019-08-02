import plotly.figure_factory as ff
import plotly.graph_objects as go


def visualize_2d_pmf(data, x_ticks, y_ticks, x_label, y_label, title):
    fig = ff.create_annotated_heatmap(z=data, x=x_ticks, y=y_ticks, colorscale='Greys')
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    fig.update_layout(title=title, autosize=False, width=400, height=400)
    return fig


def visualize_1d_pm(data, names, x_label, y_label, title):
    fig = go.Figure(data=[
        go.Bar(x=names, y=data)
    ])
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    fig.update_layout(title=title, autosize=False, width=500, height=400)
    return fig



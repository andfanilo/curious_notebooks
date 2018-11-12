import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash()
INPUT_ID = 'my-id'
OUTPUT_ID = 'my-div'

app.css.append_css({
    "external_url": "https://stackpath.bootstrapcdn.com/bootstrap/4.1.0/css/bootstrap.min.css"
})

app.layout = html.Div([
    html.H1(children='Hello world'),
    dcc.Input(id=INPUT_ID, value='initial value', type='text'),
    html.Div(id=OUTPUT_ID)
])


@app.callback(
    Output(component_id=OUTPUT_ID, component_property='children'),
    [Input(component_id=INPUT_ID, component_property='value')]
)
def update_output_div(input_value):
    return 'You\'ve entered "{}"'.format(input_value)


if __name__ == '__main__':
    app.run_server(debug=True)

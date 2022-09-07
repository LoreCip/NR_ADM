import sys
import h5py

import numpy as np

import dash
import dash.html as html
import dash.dcc as dcc
from dash.dependencies import Input, Output

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

## RECOVER DATA

name = sys.argv[1]
N = int(sys.argv[2])
tmax = int(sys.argv[3])
dR = float(sys.argv[4])
r = np.array([(j - 0.5)*dR for j in range(1,N+1)], dtype = np.float64)

dataH5 = h5py.File(name,'r')
data = np.zeros((len(dataH5), 7, N))
for i in range(tmax+1):
    data[i,:,:] = dataH5[f'{i}']
    
horizonData = dataH5['Horizon']

## SETUP DASH

app = dash.Dash()   #initialising dash app

def maps(string_val):
	mapps = {
		'A' : 0, 'B' : 1,
		'DA': 2, 'DB': 3,
		'KA': 4, 'KB': 5,
		'alpha': 6
	}
	return mapps[string_val]

## DEFINE PLOTS

@app.callback(Output(component_id='field_plot', component_property= 'figure'),
              [Input(component_id='dropdown', component_property= 'value'), Input('time_slider', 'value')])
def dynam_plots(dropdown_value, slider_value):
     
     idG = maps(dropdown_value)
     idT = slider_value
     
     fig = go.Figure([go.Scatter(x = r, y = data[idT,idG,:], line = dict(color = 'firebrick', width = 4), name = dropdown_value)])
     fig.update_layout(title = f'Time {idT}', xaxis_title = 'Position')
     return fig  

def horizon_plot():
    
    fig = make_subplots(rows=1, cols=4)
    fig.add_trace(
        go.Scatter(x = horizonData[:,0], y = horizonData[:,1],
                  line = dict(color = 'SlateGrey', width = 4), name = 'Isotropic Coordinates'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x = horizonData[:,0], y = horizonData[:,2],
                  line = dict(color = 'SteelBlue', width = 4), name = 'Schwarzschild Coordinates'),
        row=1, col=3
        )
    fig.add_trace(
        go.Scatter(x = horizonData[:,0], y = horizonData[:,3],
                  line = dict(color = 'Thistle', width = 4), name = 'Surface', ),
        row=1, col=4
        )
    
    fig.update_layout(title = f'Horizon', xaxis_title = 'Time')
    return fig  

##   PAGE LAYOUT

children = [
    # Title
       html.H1(id = 'H1', 
           children = 'Numerical Relativity Exercise', 
           style = {'textAlign':'center', 'marginTop':40,'marginBottom':40,'font-family':'Papyrus'}),
    # Field dropdown
       dcc.Dropdown( id = 'dropdown', value = 'A',
           options = [
                       {'label': 'A', 'value':'A'},
                       {'label': 'B', 'value':'B'},
                       {'label': 'DA', 'value':'DA'},
                       {'label': 'DB', 'value':'DB'},
                       {'label': 'KA', 'value':'KA'},
                       {'label': 'KB', 'value':'KB'},
                       {'label': r'\alpha', 'value':'alpha'}
                       ],
           style = {'textAlign':'center', 'marginTop':40,'marginBottom':40}),
    # Plot
       dcc.Graph(id = 'field_plot'),
    # Time slider
       dcc.Slider(0, tmax, 1,
                   value=10,
                   id='time_slider',
                   marks=None,
                   tooltip={"placement": "bottom", "always_visible": False}),
    # Plot
       dcc.Graph(id = 'horizon_plot', figure = horizon_plot())
]


## SETUP DASH


app.layout = html.Div(id = 'parent', style = {'background-color': 'Beige'}, children = children)

if __name__ == '__main__':
    app.run_server(port = 8880)
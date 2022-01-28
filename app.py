import pandas as pd
import plotly.graph_objects as go
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output

model=joblib.load('model_dt_random_up')
employee = pd.read_csv("data_train.csv")
employee_test = pd.read_csv("data_test.csv")

cat_cols = ["gender","relevent_experience","education_level","major_discipline","experience","last_new_job"]

employee[cat_cols] = employee.select_dtypes("object").astype("category")
employee['target'] = employee['target'].astype("category")
employee_test[cat_cols] = employee_test.select_dtypes("object").astype("category")
employee_test['target'] = employee_test['target'].astype("category")

X_train = employee.drop("target", axis=1)
y_train = employee["target"]

X_test = employee_test.drop("target", axis=1)
y_test = employee_test["target"]

encoder = OneHotEncoder(drop='first', sparse=False)
encoder.fit(X_train[cat_cols])

X_train_encoded = pd.DataFrame(encoder.transform(X_train[cat_cols]),
                               index=X_train.index,
                               columns=encoder.get_feature_names(X_train[cat_cols].columns))
X_train_dummy = pd.concat([X_train.select_dtypes(exclude='category'),
                           X_train_encoded], axis=1)

X_test_encoded = pd.DataFrame(encoder.transform(X_test[cat_cols]),
                               index=X_test.index,
                               columns=encoder.get_feature_names(X_test[cat_cols].columns))
X_test_dummy = pd.concat([X_test.select_dtypes(exclude='category'),
                           X_test_encoded], axis=1)


# input prediction
baris=0

# Card Contents
y_pred_dt_test_up = model.predict(X_test_dummy)
acc_score = ((accuracy_score(y_true=y_test, y_pred=y_pred_dt_test_up))*100).round(2)
pred_score = ((precision_score(y_true=y_test, y_pred=y_pred_dt_test_up, pos_label=1))*100).round(2)
recall = ((recall_score(y_true=y_test, y_pred=y_pred_dt_test_up, pos_label=1))*100).round(2)


card_accuracy = [
    dbc.CardHeader("Accuracy Score"),
    dbc.CardBody(
        [
            html.H1(
                f"{acc_score}%",
                className="card-text",
            ),
        ]
    ),
]

card_precision = [
    dbc.CardHeader("Precision Score"),
    dbc.CardBody(
        [
            html.H1(
                f"{pred_score}%",
                className="card-text",
            ),
        ]
    ),
]

card_recall = [
    dbc.CardHeader("Recall Score"),
    dbc.CardBody(
        [
            html.H1(
                f"{recall}%",
                className="card-text",
            ),
        ]
    ),
]

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
app.title = 'Predictive Analytics Dashboard'
server = app.server

app.layout = html.Div(children=[
    dbc.NavbarSimple(
        children=[html.H1('Predictive Analytics Dashboard'),
        ],
        brand="",
        brand_href="#",
        color="maroon",
    ),
    html.Br(),
    dbc.Row(
        dbc.Alert(id='tbl_out', color='dark'),
    ),
    dbc.Row([
        dbc.Col([
            html.Br(), html.Br(),
            dbc.Row(
                dash_table.DataTable(
                id='table_data_test',
                columns=[{'name': i, 'id': i} for i in X_test.columns],
                data=X_test.to_dict('records'),
                fixed_rows={'headers': True},
                style_cell={
                    'minWidth':80, 'maxWidth': 300, 'width': 110
                },
                style_header={
                    'backgroundColor': 'maroon',
                    'color': 'black'
                },
                style_data={
                    'backgroundColor': 'darkgray',
                    'color': 'black'
                }

        ),
            )
        ], width=6),
        # dbc.Col(dcc.Graph(id='prediction'), width=6), 
        dbc.Col(
            [
                dbc.Row(
                    dcc.Graph(id='prediction',
                            figure=go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = model.predict_proba(X_test_dummy.iloc[[baris],:])[0][1],
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Probability"},
                            delta = {'reference': 1},
                            gauge = {'axis': {'range': [0,1]},
                                    'bar': {'color': 'maroon'},
                                    'borderwidth': 2,
                                    'bordercolor': 'black',
                                    'steps': [
                                        {'range': [0, 0.5], 'color':'darkgray'},
                                        {'range': [0.5, 1], 'color': 'gray'}
                                    ],
                                    'threshold': {
                                        'line': {'color': 'red', 'width': 2},
                                        'thickness': 0.9,
                                        'value': 0.5
                                    }
                                    
                                    }
                        )),
            )
                ),
                dbc.Row([
                    dbc.Col(dbc.Card(card_accuracy, color="darkgray", inverse=True)),
                    dbc.Col(dbc.Card(card_precision, color="gray", inverse=True)),
                    dbc.Col(dbc.Card(card_recall, color="maroon", inverse=True)),
                ]),
                html.Br(),
                # dbc.Row(
                #     dbc.Alert(id='tbl_out', color='dark'),
                # )
            ], width=6), 
])
])

@app.callback(
    Output('tbl_out', 'children'),
    Input('table_data_test', 'active_cell')
    )

def update_row(active_cell):
    if active_cell:
        employee_id = X_test.iloc[active_cell['row'],0]
        prob = ((model.predict_proba(X_test_dummy.iloc[[active_cell['row']],:])[0][1])*100).round(1)
        return f"Employee with ID {employee_id} has probability of {prob}% to be promoted"

    else:
        return "Please select the row you want to predict."
    # return str(active_cell['row']) if active_cell else "Click the table"

@app.callback(
    Output('prediction', 'figure'),
    Input('table_data_test', 'active_cell')
    )

def update_graphs(active_cell):
    row_baris = int(active_cell['row'])
    return go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = model.predict_proba(X_test_dummy.iloc[[row_baris],:])[0][1],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probability"},
                delta = {'reference': 1},
                gauge = {'axis': {'range': [0,1]},
                        'bar': {'color': 'maroon'},
                        'borderwidth': 2,
                        'bordercolor': 'black',
                        'steps': [
                            {'range': [0, 0.5], 'color':'darkgray'},
                            {'range': [0.5, 1], 'color': 'gray'}
                        ],
                        'threshold': {
                            'line': {'color': 'red', 'width': 2},
                            'thickness': 0.9,
                            'value': 0.5
                        }
                        
                        }
            ))
    
    
    

if __name__ == "__main__":
    app.run_server()
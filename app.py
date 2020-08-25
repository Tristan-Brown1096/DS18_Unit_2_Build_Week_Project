import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from xgboost import XGBClassifier
import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.pipeline import make_pipeline

df = pd.read_csv('https://raw.githubusercontent.com/Tristan-Brown1096/DS18_Unit_2_Build_Week_Project/master/data/cbb.csv')
df=df[df['POSTSEASON']!='no tourney']
df.TEAM = [team.replace(' ', '') for team in df.TEAM]
train = df[df['YEAR'] <= 2017]
val = df[df['YEAR'] == 2018]
test = df[df['YEAR'] == 2019]
train['YEAR'] = train['YEAR'].astype(str)
val['YEAR'] = val['YEAR'].astype(str)
test['YEAR'] = test['YEAR'].astype(str)
train['ID'] = train.TEAM + train.YEAR
val['ID'] = val.TEAM + val.YEAR
test['ID'] = test.TEAM + test.YEAR
train = train.set_index('ID')
val = val.set_index('ID')
test = test.set_index('ID')
target = 'POSTSEASON'
X_train = train.drop([target, 'TEAM', 'YEAR', 'W', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
y_train = train[target]

X_val = val.drop([target, 'TEAM', 'YEAR', 'W', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
y_val = val[target]

X_test = test.drop([target, 'TEAM', 'YEAR', 'W', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
y_test = test[target]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(['Conference:', dcc.Input(id='Conference', type='text', value='SEC')]),
    html.Div(['Games Played:',dcc.Input(id='Games', type='text', value=31)]),
    html.Div(['Adjusted Offensive Efficiency:',dcc.Input(id='Adjusted-Offensive-Efficiency', type='text', value=111.1)]),
    html.Div(['Adjusted Defensive Efficiency:',dcc.Input(id='Adjusted-Defensive-Efficiency', type='text', value=96.1)]),
    html.Div(['Barthag(likelihood to beat average D1 team):',dcc.Input(id='Barthag', type='text', value=0.8416)]),
    html.Div(['Effective FG%:',dcc.Input(id='Effective-FG%', type='text', value=50)]),
    html.Div(['Effective Defensive FG%:',dcc.Input(id='Effective-Defensive-FG%', type='text', value=47.1)]),
    html.Div(['Offensive Turnover Rate:',dcc.Input(id='Offensive-Turnover-Rate', type='text', value=17.9)]),
    html.Div(['Defensive Turnover Rate:',dcc.Input(id='Defensive-Turnover-Rate', type='text', value=18.7)]),
    html.Div(['Offensive Rebound Rate:',dcc.Input(id='Offensive-Rebound-Rate', type='text', value=35.3)]),
    html.Div(['Defensive Rebound Rate:',dcc.Input(id='Defensive-Rebound-Rate', type='text', value=27.5)]),
    html.Div(['Offensive Free Throw Rate:',dcc.Input(id='Offensive-Free-Throw-Rate', type='text', value=43.8)]),
    html.Div(['Defensive Free Throw Rate:',dcc.Input(id='Defensive-Free-Throw-Rate', type='text', value=33.9)]),
    html.Div(['2-Point Shot%:',dcc.Input(id='2-Point-Shot%', type='text', value=53.3)]),
    html.Div(['Defensive 2-Point Shot%:',dcc.Input(id='Defensive-2-Point-Shot%', type='text', value=46.2)]),
    html.Div(['3-Point Shot%:',dcc.Input(id='3-Point-Shot%', type='text', value=30.6)]),
    html.Div(['Defensive 3-Point Shot%:',dcc.Input(id='Defensive-3-Point-Shot%', type='text', value=32.5)]),
    html.Div(['Adjusted Tempo:',dcc.Input(id='Adjusted-Tempo', type='text', value=69.3)]),
    html.Div(['Wins Above Bubble:',dcc.Input(id='Wins-Above-Bubble', type='text', value=4.6)]),
    html.Div(['Seed:', dcc.Input(id='Seed', type='text', value=8)]),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.Div(id='output-state' ,children='predict')
])


@app.callback(Output('output-state', 'children'),
              [Input('submit-button-state', 'n_clicks')],
              [State('Conference', 'value'),
               State('Games', 'value'),
               State('Adjusted-Offensive-Efficiency', 'value'),
               State('Adjusted-Defensive-Efficiency', 'value'),
               State('Barthag', 'value'),
               State('Effective-FG%', 'value'),
               State('Effective-Defensive-FG%', 'value'),
               State('Offensive-Turnover-Rate', 'value'),
               State('Defensive-Turnover-Rate', 'value'),
               State('Offensive-Rebound-Rate', 'value'),
               State('Defensive-Rebound-Rate', 'value'),
               State('Offensive-Free-Throw-Rate', 'value'),
               State('Defensive-Free-Throw-Rate', 'value'),
               State('2-Point-Shot%', 'value'),
               State('Defensive-2-Point-Shot%', 'value'),
               State('3-Point-Shot%', 'value'),
               State('Defensive-3-Point-Shot%', 'value'),
               State('Adjusted-Tempo', 'value'),
               State('Wins-Above-Bubble', 'value'),
               State('Seed', 'value')])
               



def predict(submit,input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, input16, input17, input18, input19, input20):
    team = {'CONF':input1, 'G':float(input2), 'ADJOE':float(input3), 'ADJDE':float(input4), 'BARTHAG':float(input5), 
        'EFG_O':float(input6), 'EFG_D':float(input7), 'TOR':float(input8), 'TORD':float(input9), 'ORB':float(input10),
        'DRB':float(input11), 'FTR':float(input12), 'FTRD':float(input13), '2P_O':float(input14), '2P_D':float(input15),
        '3P_O':float(input16), '3P_D':float(input17),'ADJ_T':float(input18), 'WAB':float(input19), 'SEED':float(input20)}
    model = make_pipeline(OrdinalEncoder(), XGBClassifier(max_depth=5, learning_rate=0.001, n_estimators=500, n_jobs=-1, objective='multi:sotmax', eval_metric='merror', num_class=8, critereon = 'entropy'))
    model.fit(X_train, y_train)    
    return model.predict([team])


if __name__ == '__main__':
    app.run_server(debug=True)
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
from plotly.colors import n_colors
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import warnings
import os

warnings.filterwarnings("ignore")
warnings.warn("this will not show")

# Initialize the Dash app with Bootstrap dark theme and custom styles
app = dash.Dash(__name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP, "https://use.fontawesome.com/releases/v5.15.4/css/all.css"],
    suppress_callback_exceptions=True,
    title='Sleep Health Dashboard',
    update_title=None
)

# Meta tags for proper scaling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600&display=swap" rel="stylesheet">
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Load the dataset
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("No Disorder")
df["BMI Category"][df["BMI Category"]=="Normal Weight"] = "Normal"

# Create the navigation bar
navbar = html.Nav(
    html.Div([
        html.Div([
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Distributions", href="/distributions", active="exact")),
                    dbc.NavItem(dbc.NavLink("Features Correlations", href="/correlation-inputs", active="exact")),
                    dbc.NavItem(dbc.NavLink("Targets Correlations", href="/correlation-targets", active="exact")),
                    dbc.NavItem(dbc.NavLink("Features/Targets Relations", href="/correlation-all", active="exact")),
                ],
                navbar=True,
                className="nav-links",
            ),
        ], className="nav-center"),
        html.Button(
            html.I(className="fas fa-moon"),
            id="theme-toggle",
            className="theme-toggle",
        ),
    ], className="nav-content"),
    className="top-nav",
)

# Theme store
theme_store = dcc.Store(id='theme-store', data={'theme': 'light'})

# Define the layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    theme_store,
    html.Div([
        navbar,
        html.Div(
            html.Div(
                id='page-content',
                className='page-content'
            ),
            className="content-wrapper"
        )
    ], 
    id='app-container',
    className='app-container')
])

# Theme switching callback
@app.callback(
    [Output('theme-store', 'data'),
     Output('app-container', 'className')],
    [Input('theme-toggle', 'n_clicks')],
    [State('theme-store', 'data')]
)
def toggle_theme(n_clicks, theme_data):
    if n_clicks is None:
        return theme_data, 'app-container'
    
    new_theme = 'dark' if theme_data['theme'] == 'light' else 'light'
    return {'theme': new_theme}, f'app-container theme-{new_theme}'

# Icon toggle callback
@app.callback(
    Output('theme-toggle', 'children'),
    [Input('theme-store', 'data')]
)
def update_icon(theme_data):
    icon_class = "fas fa-sun" if theme_data['theme'] == 'dark' else "fas fa-moon"
    return html.I(className=icon_class)

def get_theme_colors(theme_data):
    """Get theme-specific colors for plots"""
    if theme_data and theme_data.get('theme') == 'dark':
        return {
            'bg': '#332B4A',
            'text': '#E2DFF0',
            'grid': '#483A6F',
            'bar': '#9575CD'
        }
    return {
        'bg': '#FFFFFF',
        'text': '#3D2B61',
        'grid': '#D1C4E9',
        'bar': '#9C6CD4'
    }

def create_graph_card(figure, className='distribution-graph', is_wide=False):
    """Create a card containing a graph"""
    return dbc.Card(
        dbc.CardBody(
            dcc.Graph(
                figure=figure,
                className=className,
                config={
                    'displayModeBar': False,
                    'responsive': True,
                    'staticPlot': False
                },
                style={
                    'width': '100%',
                    'height': '100%'
                }
            ),
            style={
                'height': '100%',
                'padding': '0.5rem'
            }
        ),
        className=f'graph-card{" wide-card" if is_wide else ""}'
    )
def create_distributions_page(theme_data):
    """Create the distributions page with all its figures"""
    colors = get_theme_colors(theme_data)

    # Gender Distribution
    gender_counts = df['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    fig_gender = px.pie(
        gender_counts,
        names="Gender",
        values="Count",
        color="Gender",
        color_discrete_sequence=['#311b92', '#dcd0ff']
    )
    fig_gender.update_layout(
        title=dict(
            text="<b>Gender Distribution</b>",
            x=0.5,
            xanchor='center'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        )
    )

    # Age Distribution
    fig_age = px.histogram(
        data_frame=df,
        x="Age",
        nbins=20,
        color_discrete_sequence=['#311b92']
    )
    fig_age.update_layout(
        title=dict(
            text="<b>Age Distribution</b>",
            x=0.5,
            xanchor='center'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        )
    )

    # BMI Categories
    bmi_counts = df["BMI Category"].value_counts().reset_index()
    bmi_counts.columns = ["BMI Category", "Count"]
    fig_bmi = px.pie(
        bmi_counts,
        names="BMI Category",
        values="Count",
        color="BMI Category",
        color_discrete_sequence=['#a066c2', '#dcd0ff', '#311b92']
    )
    fig_bmi.update_layout(
        title=dict(
            text="<b>BMI Categories Distribution</b>",
            x=0.5,
            xanchor='center'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        )
    )

    # Sleep Quality Distribution
    sleep_quality_distribution = df['Quality of Sleep'].value_counts().sort_index().reset_index()
    sleep_quality_distribution.columns = ['Quality of Sleep', 'Count']
    fig_sleep_quality = px.bar(
        sleep_quality_distribution,
        x='Count',
        y='Quality of Sleep',
        orientation='h',
        color='Quality of Sleep',
        color_continuous_scale=px.colors.sequential.Purpor
    )
    fig_sleep_quality.update_layout(
        title=dict(
            text="<b>Sleep Quality Distribution</b>",
            x=0.5,
            xanchor='center'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        )
    )

    # Sleep Disorders funnel chart
    sleep_disorder_counts = df["Sleep Disorder"].fillna("None").value_counts().reset_index()
    sleep_disorder_counts.columns = ["Sleep Disorder", "Count"]
    fig_sleep_disorders = px.funnel(
        sleep_disorder_counts,
        x="Count",
        y="Sleep Disorder",
        title="<b>Distribution of Sleep Disorders<b>",
        color_discrete_sequence=['#311b92']
    )
    fig_sleep_disorders.update_layout(
        title=dict(
            text="<b>Sleep Disorders Distribution</b>",
            x=0.5,
            xanchor='center'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        )
    )

    # Stress Levels Distribution
    stress_counts = df["Stress Level"].value_counts().sort_index().reset_index()
    stress_counts.columns = ["Stress Level", "Count"]
    fig_stress = px.bar(
        stress_counts,
        x="Stress Level",
        y="Count",
        color="Stress Level",
        color_continuous_scale=px.colors.sequential.Purpor
    )
    fig_stress.update_layout(
        title=dict(
            text="<b>Distribution of Stress Levels</b>",
            x=0.5,
            xanchor='center'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        )
    )

    # Daily Steps Distribution with KDE
    fig_steps = px.histogram(
        df,
        x="Daily Steps",
        nbins=10,
        color_discrete_sequence=['#dcd0ff'],
        title="Distribution of Daily Steps"
    )
    fig_steps.update_layout(
        title=dict(
            text="<b>Distribution of Daily Steps</b>",
            x=0.5,
            xanchor='center'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        )
    )

    # Add KDE line and update x-axis range
    x_vals = np.linspace(df["Daily Steps"].min(), 11000, 200)  # Extend range to 11000
    kde = gaussian_kde(df["Daily Steps"])
    density = kde(x_vals)
    # Calculate bin width
    bin_width = (df["Daily Steps"].max() - df["Daily Steps"].min()) / 10  # 10 bins
    # Scale the KDE to match histogram frequencies
    density_scaled = density * len(df["Daily Steps"]) * bin_width

    # Update x-axis range
    fig_steps.update_xaxes(
        range=[df["Daily Steps"].min(), 11000],  # Extend range
        tickvals=np.arange(df["Daily Steps"].min(), 11001, step=500),
        tickfont=dict(size=12)
    )

    # Add KDE line
    fig_steps.add_scatter(
        x=x_vals,
        y=density_scaled,
        mode='lines',
        name='KDE',
        line=dict(color='#311b92')
    )

    # Process category metrics and occupation data
    occupation_categories = {
        'Healthcare': ['Doctor', 'Nurse'],
        'Technology': ['Software Engineer', 'Engineer', 'Scientist'],
        'Business': ['Accountant', 'Manager'],
        'Sales': ['Salesperson', 'Sales Representative'],
        'Education/Legal': ['Teacher', 'Lawyer']
    }

    # Create mapping from occupation to category
    occupation_to_category = {}
    for category, occupations in occupation_categories.items():
        for occupation in occupations:
            occupation_to_category[occupation] = category

    # Calculate health metrics by category first
    category_metrics = {}
    for occupation in df['Occupation'].unique():
        subset = df[df['Occupation'] == occupation]
        category = occupation_to_category.get(occupation, 'Other')
        
        if category not in category_metrics:
            category_metrics[category] = {
                'Sleep Quality': [],
                'Sleep Duration': [],
                'Physical Activity': [],
                'Stress Level': []
            }
        
        category_metrics[category]['Sleep Quality'].extend(subset['Quality of Sleep'].tolist())
        category_metrics[category]['Sleep Duration'].extend(subset['Sleep Duration'].tolist())
        category_metrics[category]['Physical Activity'].extend(subset['Physical Activity Level'].tolist())
        category_metrics[category]['Stress Level'].extend(subset['Stress Level'].tolist())

    # Create sleep categories
    df['Sleep Duration Category'] = pd.cut(
        df['Sleep Duration'],
        bins=[0, 6.0, 7.0, 10],
        labels=['Short (<6h)', 'Average (6-7h)', 'Long (>7h)']
    )

    df['Sleep Quality Category'] = pd.cut(
        df['Quality of Sleep'],
        bins=[0, 4, 7, 10],
        labels=['Poor', 'Average', 'Excellent']
    )

    # Categories to visualize
    categories = ['Occupation', 'BMI Category', 'Sleep Disorder',
                'Sleep Duration Category', 'Sleep Quality Category']

    fig_health_metrics = go.Figure()

    # Color palettes with specific colorscales for each category
    palettes = {
        'Occupation': px.colors.sequential.Agsunset,
        'BMI Category': px.colors.sequential.Sunsetdark,
        'Sleep Disorder': px.colors.sequential.Purp,
        'Sleep Duration Category': px.colors.sequential.Sunset,
        'Sleep Quality Category': px.colors.sequential.Purp_r
    }

    # Custom multipliers for each category
    category_multipliers = {
        'Occupation': 0.65,
        'BMI Category': 1.23,
        'Sleep Disorder': 1.14,
        'Sleep Duration Category': 1.16,
        'Sleep Quality Category': 1.25
    }

    # Define explicit ordered categories for ordinal variables
    ordered_categories = {
        'Sleep Duration Category': ['Short (<6h)', 'Average (6-7h)', 'Long (>7h)'],
        'Sleep Quality Category': ['Poor', 'Average', 'Excellent'],
        'BMI Category': ['Underweight', 'Normal', 'Overweight', 'Obese']
    }

    # Process each category
    for category in categories:
        # Get value counts and ensure Sleep Disorder shows all values
        if category == 'Sleep Disorder':
            all_values = df[category].unique()
            value_counts = df[category].value_counts().reindex(all_values).fillna(0)
        else:
            value_counts = df[category].value_counts()

        # Handle ordering properly
        if category in ordered_categories:
            # Use predefined order for ordinal categories
            valid_categories = [cat for cat in ordered_categories[category] if cat in value_counts.index]
            value_counts = value_counts.reindex(valid_categories).dropna()
        elif category == 'Occupation' and len(value_counts) > 12:
            # Handle Occupation specially (limit to top 12)
            value_counts = value_counts.nlargest(12)
            other_count = df[category].shape[0] - value_counts.sum()
            if other_count > 0:
                value_counts['Other'] = other_count
        else:
            # For other categories, sort by count in descending order
            value_counts = value_counts.sort_values(ascending=False)

        subcategories = value_counts.index.tolist()
        counts = value_counts.values

        # Get colors from specific palette for each category
        palette = palettes.get(category)

        # Generate colors for each subcategory
        if len(subcategories) <= 1:
            colors = [palette[0]]
        else:
            if category in ordered_categories:
                indices = np.linspace(0, len(palette)-1, len(subcategories)).astype(int)
                colors = [palette[i] for i in indices]
            else:
                indices = np.linspace(0, len(palette)-1, len(subcategories)).astype(int)
                colors = [palette[i] for i in indices]

        # Transform counts for better visualization
        transformed_counts = np.power(counts, 0.5) * category_multipliers[category]

        fig_health_metrics.add_trace(
            go.Barpolar(
                r=transformed_counts,
                theta=[category] * len(subcategories),
                width=0.85,
                marker_color=colors,
                marker_line_color="white",
                marker_line_width=1,
                opacity=0.9,
                name=category,
                customdata=[[s, int(c)] for s, c in zip(subcategories, counts)],
                hovertemplate="<b>%{theta}</b><br>%{customdata[0]}<br>Count: %{customdata[1]}<extra></extra>"
            )
        )

    # Update layout with theme-aware colors and repositioned title/legend
    fig_health_metrics.update_layout(
        title=dict(
            text="<b>Health Metrics by Category</b>",
            x=0.25,  # Move title to the left (was 0.8)
            y=0.90,  # Lower the title slightly (was 0.98)
            yanchor="bottom",
            xanchor='center'
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=False,
                gridcolor="lightgrey"
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                tickfont=dict(size=14)
            )
        ),
        height=400,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom", 
            y=1.15,
            xanchor="right",  # Changed to "right"
            x=0.98,          # Changed to 0.98 for right positioning
            font=dict(size=10),
            traceorder="grouped",
            tracegroupgap=5
        ),
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        )
    )

    # Create treemap data
    treemap_data = []
    for occupation in df['Occupation'].unique():
        subset = df[df['Occupation'] == occupation]
        category = occupation_to_category.get(occupation, 'Other')
        sleep_quality = subset['Quality of Sleep'].mean()
        sleep_duration = subset['Sleep Duration'].mean()
        treemap_data.append({
            'Category': category,
            'Occupation': occupation,
            'Count': len(subset),
            'Sleep Quality': sleep_quality,
            'Sleep Duration': sleep_duration
        })

    treemap_df = pd.DataFrame(treemap_data)
    fig_occupation = px.treemap(
        treemap_df,
        path=[px.Constant('All Occupations'), 'Category', 'Occupation'],
        values='Count',
        color='Category',
        color_discrete_sequence=px.colors.sequential.Purpor,
        hover_data=['Sleep Duration', 'Sleep Quality'],
        title='Sleep Metrics by Occupation Group'
    )
    fig_occupation.update_traces(
        hovertemplate='<b>%{label}</b><br>' +
                     'Count: %{value}<br>' +
                     'Sleep Quality: %{customdata[1]:.1f}<br>' +
                     'Sleep Duration: %{customdata[0]:.1f}h<br>' +
                     '<extra></extra>',
        marker=dict(
            line=dict(
                width=1,
                color='white'
            )
        )
    )
    # Update treemap layout with theme colors
    fig_occupation.update_layout(
        title=dict(
            text="<b>Sleep Metrics by Occupation</b>",
            x=0.5,
            xanchor='center'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        ),
        height=400
    )

    # Create missing values visualization
    # Define colors and color list
    highlight_color = '#dcd0ff'
    base_color = '#311b92'
    custom_color_list = [
        highlight_color, base_color, base_color, base_color, base_color,
        base_color, base_color, base_color, highlight_color, highlight_color,
        base_color, base_color, base_color
    ]

    # Compute the number of non-missing values for each column
    completeness = df.notnull().sum()
    df_complete = pd.DataFrame({
        'Column': completeness.index,
        'NonMissing': completeness.values
    })
    df_complete = df_complete.sort_values('NonMissing', ascending=False)

    # Create color mapping
    custom_colors = []
    for i, col in enumerate(df_complete['Column']):
        if i < len(custom_color_list):
            custom_colors.append(custom_color_list[i])
        else:
            custom_colors.append(base_color)
    df_complete['Color'] = custom_colors
    color_mapping = dict(zip(df_complete['Column'], df_complete['Color']))

    # Create missing values bar chart
    fig_missing = px.bar(
        df_complete,
        x='Column',
        y='NonMissing',
        text='NonMissing',
        color='Column',
        color_discrete_map=color_mapping,
        title="<b>Visualization of All The Variables In The Dataset (374 Rows)</b>"
    )

    fig_missing.update_layout(
        plot_bgcolor='#faf8ff',
        paper_bgcolor='#faf8ff',
        title_x=0.4,
        xaxis_title="",
        yaxis_title="Non-missing Count",
        font=dict(family='Serif', size=12),
        margin=dict(l=40, r=40, t=80, b=80),
        height=400
    )

    fig_missing.update_xaxes(
        tickangle=50,
        tickfont=dict(family='Serif', color='#512b58')
    )

    fig_missing.update_yaxes(showticklabels=False)

    # Create regular cards
    cards = []
    # First row (full width missing values chart)
    cards.append(create_graph_card(fig_missing, className='distribution-graph', is_wide=True))
    # Second row
    cards.extend([
        create_graph_card(fig_gender),
        create_graph_card(fig_age),
    ])
    # Third row
    cards.extend([
        create_graph_card(fig_bmi),
        create_graph_card(fig_sleep_disorders),
    ])
    # Fourth row
    cards.extend([
        create_graph_card(fig_stress),
        create_graph_card(fig_steps),
    ])
    # Fifth row
    cards.extend([
        create_graph_card(fig_sleep_quality),
        create_graph_card(fig_health_metrics)
    ])
    # Last row (full width treemap)
    cards.append(create_graph_card(fig_occupation, className='distribution-graph', is_wide=True))
    
    return html.Div([
        html.H1('Distributions', style={'textAlign': 'center'}),
        html.P('Explore the distributions of the dataset variables.', className='subtitle', style={'textAlign': 'center'}),
        html.Div(cards, className='graph-container')
    ])
    
    
def create_correlation_targets_page(theme_data):
    """Create the input-target relations page with all its figures"""
    colors = get_theme_colors(theme_data)
    
    # 1. Sunburst Chart - Sleep Disorder effect on Sleep Duration
    fig_sunburst = px.sunburst(
        df,
        path=[px.Constant('Sleep quality'), 'Sleep Disorder', 'Quality of Sleep'],
        values='Sleep Duration',
        color='Sleep Disorder',
        color_discrete_map={
            '(?)': '#faf8ff',
            'Insomnia': '#311b92',
            'Sleep Apnea': '#a066c2',
            'No Disorder': '#dcd0ff'
        }
    )

    fig_sunburst.update_layout(
        title=dict(
            text="<b>Effect of sleep disorder on sleep duration</b>",
            x=0.5,
            xanchor='center',
            y=0.98,  # Positioned higher
            font=dict(size=14)  # Smaller font
        ),
        margin=dict(l=20, r=20, t=60, b=20),  # Increased top margin
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        )
    )
    
    # 2. Sleep Duration vs Quality Chart (Combined Line and Bubble)
    # Calculate the number of people for each sleep duration
    bubble_size = df['Sleep Duration'].value_counts().reset_index()
    bubble_size.columns = ['Sleep Duration', 'Count']

    # Group by sleep duration to calculate the average sleep quality
    data_grouped = df.groupby("Sleep Duration")["Quality of Sleep"].mean().reset_index()

    # Create the line and bubble chart
    fig_sleep_quality = go.Figure()
    
    # Create the line trace for average sleep quality
    fig_sleep_quality.add_trace(go.Scatter(
        x=data_grouped["Sleep Duration"],
        y=data_grouped["Quality of Sleep"],
        mode="lines",
        line=dict(color='#6a0dad'),
        name="Average Sleep Quality"
    ))

    # Create the bubble trace
    fig_sleep_quality.add_trace(go.Scatter(
        x=bubble_size["Sleep Duration"],
        y=data_grouped["Quality of Sleep"],
        mode="markers",
        marker=dict(
            size=bubble_size["Count"],
            color='rgba(155, 89, 182, 0.7)',
            line=dict(color='#8e44ad', width=1)
        ),
        name="Number of People by Sleep Duration"
    ))

    # Update layout for theme compatibility
    fig_sleep_quality.update_layout(
        title=dict(
            text="<b>Sleep Quality by Sleep Duration</b>",  # Shortened title
            x=0.5,
            xanchor='center',
            y=0.98,  # Positioned higher
            font=dict(size=14)  # Smaller font
        ),
        xaxis_title="Sleep Duration (hours)",
        yaxis_title="Sleep Quality",
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        ),
        margin=dict(l=20, r=20, t=60, b=20),  # Increased top margin
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        )
    )
    
    # 3. Stress Level vs Sleep Duration Box Plot
    stress_levels = sorted(df["Sleep Disorder"].unique(), reverse=True)
    
    # Generate colors from reversed purple palette
    num_levels = len(stress_levels)
    stress_colors = px.colors.sample_colorscale("Purp_r", [i/(num_levels-1) for i in range(num_levels)])
    
    fig_stress_box = go.Figure()
    
    # Add box traces for each stress level
    for i, stress in enumerate(stress_levels):
        subset = df[df["Sleep Disorder"] == stress]

        fig_stress_box.add_trace(go.Box(
            x=[stress] * len(subset),
            y=subset["Sleep Duration"],
            name=str(stress),
            fillcolor=stress_colors[i],
            line=dict(color="black", width=1.5),
            opacity=0.9,
            boxmean=True,
            boxpoints=False,
            width=0.8,
            marker=dict(
                color=stress_colors[i],
                opacity=0.7
            ),
            whiskerwidth=0.8,
            line_width=1.5
        ))
    
    # Style the plot with dashboard theme compatibility
    fig_stress_box.update_layout(
        title=dict(
            text="<b>Impact of sleep disorder on sleep duration</b>",
            x=0.5,
            xanchor='center',
            y=0.98,  # Positioned higher
            font=dict(size=14)  # Smaller font
        ),
        xaxis=dict(
            title="Sleep Disorder",
            categoryorder='array',
            categoryarray=stress_levels,
            showgrid=True,
            showline=True,
            linewidth=1,
            gridcolor="rgba(0,0,0,0.05)",
            linecolor='rgba(0,0,0,0.2)',
            mirror=True
        ),
        yaxis=dict(
            title="Sleep Duration (hours)",
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)',
            mirror=True
        ),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        showlegend=False,
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        ),
        margin=dict(t=60, b=20, l=20, r=20)  # Increased top margin
    )
    
    # Create cards with only 3 visualizations in a 2x2 grid
    cards = [
        create_graph_card(fig_sunburst, is_wide=True),
        create_graph_card(fig_sleep_quality),
        create_graph_card(fig_stress_box),
        # Fourth card removed as requested
    ]
    
    return html.Div([
        html.H1('Targets Correlations', style={'textAlign': 'center'}),
        html.P('Explore the correlations between target variables.', 
               className='subtitle', 
               style={'textAlign': 'center'}),
        html.Div(cards, className='graph-container')
    ])
    
def create_correlation_inputs_page(theme_data):
    """Create the correlations inputs page with figures showing relationships between input variables"""
    
    # Set font color based on theme
    font_color = '#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
    
    # 1. Age vs Stress Level KDE plot
    # Create stress categories if not already in df
    df_copy = df.copy()
    conditions = [
        (df_copy['Stress Level'] == 3) | (df_copy['Stress Level'] == 4),
        (df_copy['Stress Level'] == 5) | (df_copy['Stress Level'] == 6),
        (df_copy['Stress Level'] == 7) | (df_copy['Stress Level'] == 8)
    ]
    choices = ['Low Stress', 'Medium Stress', 'High Stress']
    
    df_copy['Stress Category'] = np.select(conditions, choices, default='Unknown')
    
    # Define colors with transparency included
    kde_colors = {
        "Low Stress": "rgba(0, 32, 76, 0.5)",    # #00204c with 0.5 alpha
        "Medium Stress": "rgba(123, 109, 155, 0.5)",  # #7b6d9b with 0.5 alpha
        "High Stress": "rgba(255, 110, 84, 0.5)"      # #ff6e54 with 0.5 alpha
    }
    
    # Create figure
    fig_age_stress = go.Figure()
    
    # Calculate KDE for each category
    x_range = np.linspace(df_copy['Age'].min() - 2, df_copy['Age'].max() + 2, 300)
    
    for category in choices:
        subset = df_copy[df_copy['Stress Category'] == category]['Age']
        if len(subset) > 1:  # Need at least 2 points for KDE
            kde = gaussian_kde(subset)
            y_values = kde(x_range)
            
            # Add KDE curves with proper alpha transparency
            fig_age_stress.add_trace(go.Scatter(
                x=x_range,
                y=y_values,
                mode='lines',
                line=dict(width=2, color=kde_colors[category].replace("0.5)", "1.0)")),  # Solid line
                name=category,
                fill='tozeroy',
                fillcolor=kde_colors[category]  # Fill with transparency
            ))
    
    fig_age_stress.update_layout(
        title=dict(
            text='<b>Age vs Stress Level</b>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Age of the Person',
        yaxis_title='Density',
        xaxis=dict(range=[df_copy['Age'].min() - 2, df_copy['Age'].max() + 2]),
        yaxis=dict(range=[0, None]),
        legend_title='Stress Category',
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # 2. Occupation vs Stress Level bar chart
    avg_data = df.groupby('Occupation', as_index=False)['Stress Level'].mean()
    avg_data = avg_data.sort_values('Stress Level')
    
    fig_occupation_stress = px.bar(
        avg_data,
        x='Occupation',
        y='Stress Level',
        text='Stress Level',
        color='Occupation',
        color_discrete_sequence=px.colors.sequential.Purp,
        title='<b>Relationship Between Occupation and Stress Level</b>'
    )
    
    fig_occupation_stress.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    
    fig_occupation_stress.update_layout(
        xaxis_title='Occupation',
        yaxis_title='Average Stress Level',
        xaxis_tickangle=-45,
        title_x=0.5,
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # 3. Daily Steps vs Heart Rate scatter plot with violin marginals
    fig_steps_hr = px.scatter(
        df,
        x='Daily Steps',
        y='Heart Rate',
        color='BMI Category',
        size='Daily Steps',
        marginal_x='violin',
        marginal_y='violin',
        color_discrete_sequence=['#dcd0ff', '#a066c2', '#311b92'],
        title='<b>Daily Steps vs. Heart Rate by BMI Category</b>'
    )
    
    fig_steps_hr.update_layout(
        xaxis_title='Daily Steps',
        yaxis_title='Heart Rate (bpm)',
        title_x=0.5,
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Update main plot grid lines
    fig_steps_hr.update_xaxes(showgrid=True, gridcolor='#d1c4e9', row=1, col=1)
    fig_steps_hr.update_yaxes(showgrid=True, gridcolor='#d1c4e9', row=1, col=1)
    
    # Disable grid for marginal plots
    for key in fig_steps_hr.layout:
        if key.startswith('xaxis') and key != 'xaxis':
            fig_steps_hr.layout[key].showgrid = False
        if key.startswith('yaxis') and key != 'yaxis':
            fig_steps_hr.layout[key].showgrid = False
    
    # 4. Heart Rate Variation by Age scatter plot with error bars
    grouped = df.groupby("Age")["Heart Rate"].agg(["mean", "std", "count"]).reset_index()
    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
    
    fig_hr_age = px.scatter(
        grouped,
        x="Age",
        y="mean",
        error_y="se",
        title="<b>Heart Rate Variation by Age</b>"
    )
    
    fig_hr_age.update_traces(
        mode='lines+markers',
        marker=dict(color='#a066c2'),
        line=dict(color='#a066c2')
    )
    
    fig_hr_age.update_layout(
        xaxis_title="Age",
        yaxis_title="Mean Heart Rate (bpm)",
        title_x=0.5,
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig_hr_age.update_xaxes(showgrid=True, gridcolor='rgba(209, 196, 233, 0.5)')
    fig_hr_age.update_yaxes(showgrid=True, gridcolor='rgba(209, 196, 233, 0.5)')
    
    # Create cards for all charts - two per row
    cards = [
        # First row
        create_graph_card(fig_age_stress),
        create_graph_card(fig_occupation_stress),
        
        # Second row
        create_graph_card(fig_steps_hr),
        create_graph_card(fig_hr_age)
    ]
    
    return html.Div([
           html.H1('Features Correlations', style={'textAlign': 'center'}),
           html.P('Explore correlations between key features.', 
           className='subtitle', 
           style={'textAlign': 'center'}),
        html.Div(cards, className='graph-container')
    ])
    
def create_correlation_all_page(theme_data):
    """Create the input-target relations page with all its figures"""
    colors = get_theme_colors(theme_data)
    
    # 1. Sunburst Chart - Sleep Disorder effect on Sleep Duration
    fig_sunburst = px.sunburst(
        df,
        path=[px.Constant('Sleep quality'), 'Sleep Disorder', 'Quality of Sleep'],
        values='Sleep Duration',
        color='Sleep Disorder',
        color_discrete_map={
            '(?)': '#faf8ff',
            'Insomnia': '#311b92',
            'Sleep Apnea': '#a066c2',
            'No Disorder': '#dcd0ff'
        }
    )

    fig_sunburst.update_layout(
        title=dict(
            text="<b>Effect of sleep disorder on sleep duration</b>",
            x=0.5,
            xanchor='center',
            y=0.98,  # Positioned higher
            font=dict(size=14)  # Smaller font
        ),
        margin=dict(l=20, r=20, t=60, b=20),  # Increased top margin
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        )
    )
    
    # 2. Sleep Duration vs Quality Chart (Combined Line and Bubble)
    # Calculate the number of people for each sleep duration
    bubble_size = df['Sleep Duration'].value_counts().reset_index()
    bubble_size.columns = ['Sleep Duration', 'Count']

    # Group by sleep duration to calculate the average sleep quality
    data_grouped = df.groupby("Sleep Duration")["Quality of Sleep"].mean().reset_index()

    # Create the line and bubble chart
    fig_sleep_quality = go.Figure()
    
    # Create the line trace for average sleep quality
    fig_sleep_quality.add_trace(go.Scatter(
        x=data_grouped["Sleep Duration"],
        y=data_grouped["Quality of Sleep"],
        mode="lines",
        line=dict(color='#6a0dad'),
        name="Average Sleep Quality"
    ))

    # Create the bubble trace
    fig_sleep_quality.add_trace(go.Scatter(
        x=bubble_size["Sleep Duration"],
        y=data_grouped["Quality of Sleep"],
        mode="markers",
        marker=dict(
            size=bubble_size["Count"],
            color='rgba(155, 89, 182, 0.7)',
            line=dict(color='#8e44ad', width=1)
        ),
        name="Number of People by Sleep Duration"
    ))

    # Update layout for theme compatibility
    fig_sleep_quality.update_layout(
        title=dict(
            text="<b>Sleep Quality by Sleep Duration</b>",  # Shortened title
            x=0.5,
            xanchor='center',
            y=0.98,  # Positioned higher
            font=dict(size=14)  # Smaller font
        ),
        xaxis_title="Sleep Duration (hours)",
        yaxis_title="Sleep Quality",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        ),
        margin=dict(l=20, r=20, t=60, b=20),  # Increased top margin
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        )
    )
    
    # 3. Stress Level vs Sleep Duration Box Plot
    stress_levels = sorted(df["Stress Level"].unique(), reverse=True)
    
    # Generate colors from reversed purple palette
    num_levels = len(stress_levels)
    stress_colors = px.colors.sample_colorscale("Purp_r", [i/(num_levels-1) for i in range(num_levels)])
    
    fig_stress_box = go.Figure()
    
    # Add box traces for each stress level
    for i, stress in enumerate(stress_levels):
        subset = df[df["Stress Level"] == stress]

        fig_stress_box.add_trace(go.Box(
            x=[stress] * len(subset),
            y=subset["Sleep Duration"],
            name=str(stress),
            fillcolor=stress_colors[i],
            line=dict(color="black", width=1.5),
            opacity=0.9,
            boxmean=True,
            boxpoints=False,
            width=0.8,
            marker=dict(
                color=stress_colors[i],
                opacity=0.7
            ),
            whiskerwidth=0.8,
            line_width=1.5
        ))
    
    # Style the plot with dashboard theme compatibility
    fig_stress_box.update_layout(
        title=dict(
            text="<b>Impact of stress level on sleep duration</b>",
            x=0.5,
            xanchor='center',
            y=0.98,  # Positioned higher
            font=dict(size=14)  # Smaller font
        ),
        xaxis=dict(
            title="Stress Level",
            categoryorder='array',
            categoryarray=stress_levels,
            showgrid=True,
            showline=True,
            linewidth=1,
            gridcolor="rgba(0,0,0,0.05)",
            linecolor='rgba(0,0,0,0.2)',
            mirror=True
        ),
        yaxis=dict(
            title="Sleep Duration (hours)",
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)',
            mirror=True
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        ),
        margin=dict(t=60, b=20, l=20, r=20)  # Increased top margin
    )
    
    # Create cards with only 3 visualizations in a 2x2 grid
    cards = [
        create_graph_card(fig_sunburst, is_wide=True),
        create_graph_card(fig_sleep_quality),
        create_graph_card(fig_stress_box),
        # Fourth card removed as requested
    ]
    
    return html.Div([
        html.H1('Input-Target Relations'),
        html.P('Discover how different factors impact sleep quality and duration.', className='subtitle'),
        html.Div(cards, className='graph-container')
    ])

def create_correlation_all_page(theme_data):
    """Create the correlations page with all figures showing relationships between variables"""
    colors = get_theme_colors(theme_data)
    width = 1000
    height = 600
    
    # Set font color based on theme
    font_color = '#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
    
    # 1. Correlation Matrix Heatmap
    corr_matrix = df.corr(numeric_only=True)
    fig_corr_matrix = px.imshow(
        corr_matrix,
        text_auto='.1f',
        color_continuous_scale='Purp_r',
        zmin=-1, zmax=1,
        aspect="auto"
    )
    fig_corr_matrix.update_layout(
        title=dict(
            text='<b>Correlation between different variables in the dataset</b>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Variables',
        yaxis_title='Variables',
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # 2. Sleep Duration by Age Line Chart
    grouped_data = df.groupby("Age")["Sleep Duration"].mean().reset_index()
    fig_sleep_age = px.line(
        grouped_data,
        x="Age",
        y="Sleep Duration",
        title="<b>Sleep Duration Variation by Age</b>",
        markers=True
    )
    fig_sleep_age.update_traces(
        mode='lines+markers',
        marker=dict(color='#a066c2'),
        line=dict(color='#a066c2')
    )
    fig_sleep_age.update_layout(
        xaxis_title="Age",
        yaxis_title="Average Sleep Duration",
        title_x=0.5,
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # 3. Sleep Duration by BMI Category Box Plot
    fig_bmi_sleep = px.box(
        df,
        x="BMI Category",
        y="Sleep Duration",
        title="<b>Sleep Duration by BMI Category</b>",
        color="BMI Category",
        color_discrete_sequence=['#311b92', '#a066c2', '#f2a3c3']
    )
    fig_bmi_sleep.update_layout(
        xaxis_title="BMI Category",
        yaxis_title="Sleep Duration (in hours)",
        showlegend=False,
        title_x=0.5,
        plot_bgcolor='#faf8ff',
        paper_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # 4. Sleep Duration and Heart Rate KDE Plot
    fig_sleep_hr_kde = go.Figure()
    
    try:
        x = df['Heart Rate'].values
        y = df['Sleep Duration'].values
        
        # Filter out NaN values
        valid_indices = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_indices]
        y = y[valid_indices]
        
        if len(x) > 3 and len(y) > 3:
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            
            # Create grid of points
            x_range = np.linspace(60, max(x) + 1, 100)
            y_range = np.linspace(min(y) - 0.5, max(y) + 0.5, 100)
            X, Y = np.meshgrid(x_range, y_range)
            positions = np.vstack([X.ravel(), Y.ravel()])
            
            # Evaluate kernel at grid points
            Z = kde(positions).reshape(X.shape)
            
            # Add filled contours
            fig_sleep_hr_kde.add_trace(go.Contour(
                x=x_range,
                y=y_range,
                z=Z,
                colorscale='Purples',
                contours=dict(
                    start=0,
                    end=Z.max(),
                    size=(Z.max() / 10),
                    showlabels=False
                ),
                line=dict(width=0.5),
                colorbar=dict(
                    title='Density',
                    title_side='right'
                ),
                hovertemplate='Heart Rate: %{x:.1f}<br>Sleep Duration: %{y:.1f}<br>Density: %{z:.4f}<extra></extra>'
            ))
            
            # Add scatter points with low opacity
            fig_sleep_hr_kde.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(
                    color='rgba(104, 71, 141, 0.1)',
                    size=3
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
    except Exception as e:
        print(f"Error creating KDE plot: {e}")
    
    fig_sleep_hr_kde.update_layout(
        title=dict(
            text='<b>Relationship Between Sleep Duration and Heart Rate (HRR)</b>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Heart Rate',
        yaxis_title='Sleep Duration',
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # 5. Sleep Duration by Occupation Box Plot
    fig_occupation_sleep = px.box(
        df,
        x='Occupation',
        y='Sleep Duration',
        title='<b>Relationship Between Occupation and Sleep Duration</b>',
        color='Occupation',
        color_discrete_sequence=[
            "#fab8ba", "#e4b3dd", "#d086c9", "#ba5cb6",
            "#a231a3", "#87148f", "#6a0c7a", "#4d0865",
            "#320653", "#20023f", "#10002c", "#000019"
        ]
    )
    fig_occupation_sleep.update_layout(
        xaxis_title='Occupation',
        yaxis_title='Sleep Duration (Hour)',
        title_x=0.5,
        xaxis_tickangle=45,
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # 6. Sleep Disorders by Gender Bar Chart
    sleep_disorder_gender = pd.crosstab(df['Sleep Disorder'], df['Gender'])
    sleep_disorder_df = sleep_disorder_gender.reset_index().melt(
        id_vars='Sleep Disorder',
        var_name='Gender',
        value_name='Count'
    )
    fig_disorder_gender = px.bar(
        sleep_disorder_df,
        x='Sleep Disorder',
        y='Count',
        color='Gender',
        title='<b>Sleep Disorders by Gender</b>',
        color_discrete_map={'Female': '#f2a3c3', 'Male': '#a290c8'},
        text='Count',
        barmode='group'
    )
    fig_disorder_gender.update_layout(
        xaxis_title='Sleep Disorder Type',
        yaxis_title='Count',
        legend_title='Gender',
        title_x=0.5,
        xaxis={'categoryorder': 'total descending'},
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig_disorder_gender.update_yaxes(showgrid=False)
    fig_disorder_gender.update_xaxes(showgrid=False)

    # 7. Quality of Sleep by Gender Violin Plot
    color_palette = {'Male': '#a290c8', 'Female': '#f2a3c3'}
    fig_sleep_quality_gender = px.violin(
        df,
        x='Gender',
        y='Quality of Sleep',
        color='Gender',
        color_discrete_map=color_palette,
        box=True,
        title='<b>Distribution of Quality of Sleep by Gender</b>'
    )
    fig_sleep_quality_gender.update_layout(
        xaxis_title='Gender',
        yaxis_title='Quality of Sleep',
        title_x=0.5,
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # 8. 3D Scatter Plot
    custom_purple_colors = ["#f2a3c3", "#a290c8", "#fab8ba"]
    fig_3d_scatter = px.scatter_3d(
        df,
        x='BMI Category',
        y='Blood Pressure',
        z='Heart Rate',
        color='Sleep Disorder',
        symbol='Sleep Disorder',
        color_discrete_sequence=custom_purple_colors
    )
    fig_3d_scatter.update_layout(
        title=dict(
            text='<b>The relationship between (BMI Category, Blood Pressure and Heart Rate) and their effect on Sleep Disorder</b>',
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # 9. Sleep Disorders by Physical Activity Level
    try:
        subset = df[df["Sleep Disorder"] != "None"].copy()
        subset["Physical Activity Group"] = (subset["Physical Activity Level"] // 10) * 10
        subset["Activity Range"] = subset["Physical Activity Group"].astype(str) + "-" + (subset["Physical Activity Group"] + 9).astype(str)
        
        plot_data = subset.groupby(["Activity Range", "Sleep Disorder"]).size().reset_index(name="Count")
        plot_data["Sort_Order"] = plot_data["Activity Range"].str.split("-").str[0].astype(int)
        plot_data = plot_data.sort_values("Sort_Order")
        
        fig_activity_disorder = px.bar(
            plot_data,
            x="Activity Range",
            y="Count",
            color="Sleep Disorder",
            color_discrete_map={"No Disorder": "#D4C1EC", "Insomnia": "#2E1A47", "Sleep Apnea": "#654ea3"},
            title="<b>Sleep Disorders Distribution by Physical Activity Level</b>",
            labels={"Activity Range": "Physical Activity Level (Grouped by 10s) (%)", "Count": "Number of People"},
            barmode='stack',
            category_orders={"Activity Range": plot_data["Activity Range"].unique()}
        )
        fig_activity_disorder.update_layout(
            hovermode="x unified",
            paper_bgcolor='#faf8ff',
            plot_bgcolor='#faf8ff',
            legend_title_text='Sleep Disorder',
            title_x=0.5,
            font=dict(color=font_color),
            margin=dict(l=20, r=20, t=40, b=20)
        )
    except Exception as e:
        fig_activity_disorder = go.Figure()
        fig_activity_disorder.update_layout(
            title="Error creating Sleep Disorders by Activity Level chart",
            annotations=[{"text": str(e), "showarrow": False, "font": {"color": "red"}}]
        )

    # 10. Systolic BP by Quality of Sleep Violin Plot
    try:
        df['Systolic BP'] = df['Blood Pressure'].str.split('/').str[0].astype(float)
        categories = sorted(df['Quality of Sleep'].unique())
        n_categories = len(categories)
        colors = n_colors('rgb(79, 41, 146)', 'rgb(247, 159, 121)', n_categories, colortype='rgb')
        
        fig_systolic_sleep = go.Figure()
        
        for cat, color in zip(categories, colors):
            cat_data = df.loc[df['Quality of Sleep'] == cat, 'Systolic BP']
            fig_systolic_sleep.add_trace(go.Violin(
                x=cat_data,
                name=str(cat),
                line_color=color,
                orientation='h',
                box_visible=True,
                meanline_visible=True
            ))
        
        fig_systolic_sleep.update_layout(
            title="<b>Distribution of Systolic BP by Quality of Sleep</b>",
            xaxis_title="Systolic BP",
            yaxis_title="Quality of Sleep",
            xaxis_showgrid=False,
            xaxis_zeroline=False,
            violinmode='overlay',
            title_x=0.5,
            paper_bgcolor='#faf8ff',
            plot_bgcolor='#faf8ff',
            font=dict(color=font_color),
            margin=dict(l=20, r=20, t=40, b=20)
        )
    except Exception as e:
        fig_systolic_sleep = go.Figure()
        fig_systolic_sleep.update_layout(
            title="Error creating Systolic BP chart",
            annotations=[{"text": str(e), "showarrow": False, "font": {"color": "red"}}]
        )

    # 11. Quality of Sleep by Stress Level Violin
    STRESS_MIN = 3
    STRESS_MAX = 8
    COLOR_PALETTE = "Purp"
    
    all_stress_levels = list(range(STRESS_MIN, STRESS_MAX + 1))
    num_levels = len(all_stress_levels)
    violin_colors = px.colors.sample_colorscale(COLOR_PALETTE, [i/(num_levels-1) for i in range(num_levels)])
    
    fig_stress_sleep_quality = go.Figure()
    
    for i, stress in enumerate(all_stress_levels):
        subset = df[df["Stress Level"] == stress]
        
        if len(subset) > 0:
            fig_stress_sleep_quality.add_trace(go.Violin(
                x=[stress] * len(subset),
                y=subset["Quality of Sleep"],
                name=str(stress),
                fillcolor=violin_colors[i],
                line=dict(color="black", width=1.5),
                opacity=0.9,
                side='both',
                width=0.8,
                box=dict(
                    visible=True,
                    width=0.3,
                    fillcolor='white',
                    line=dict(color='black', width=1.5)
                ),
                meanline=dict(visible=False),
                points=False,
                spanmode='soft',
                bandwidth=0.7
            ))
    
    fig_stress_sleep_quality.update_layout(
        title=dict(
            text="<b>Quality of Sleep by Stress Level</b>",
            x=0.5
        ),
        xaxis=dict(
            title=dict(text="Stress Level"),
            categoryorder='array',
            categoryarray=all_stress_levels,
            tickmode='array',
            tickvals=all_stress_levels,
            ticktext=[str(level) for level in all_stress_levels],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
        yaxis=dict(
            title=dict(text="Quality of Sleep"),
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=False
        ),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )

    # 12. Age Effect on Sleep Disorder ECDF
    fig_age_disorder = px.ecdf(
        df,
        x='Age',
        color='Sleep Disorder',
        color_discrete_sequence=['#4A235A', '#8E44AD', '#C39BD3']
    )
    fig_age_disorder.update_layout(
        title=dict(
            text='<b>The effect of Age on Sleep Disorder</b>',
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # 13. Daily Steps vs Quality of Sleep Heatmap
    fig_steps_sleep = px.density_heatmap(
        df,
        x="Daily Steps",
        y="Quality of Sleep",
        nbinsx=10,
        nbinsy=10,
        text_auto=True,
        color_continuous_scale="Purp",
        title="<b>Daily Steps vs Quality of Sleep Density Heatmap</b>"
    )
    fig_steps_sleep.update_layout(
        title_x=0.5,
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # 14. Stress Level impact on Sleep Duration Violin
    stress_levels = sorted(df["Stress Level"].unique(), reverse=True)
    num_levels = len(stress_levels)
    colors = px.colors.sample_colorscale("Purp_r", [i/(num_levels-1) for i in range(num_levels)])
    
    fig_stress_sleep_duration = go.Figure()
    
    for i, stress in enumerate(stress_levels):
        subset = df[df["Stress Level"] == stress]
        
        if len(subset) > 0:
            fig_stress_sleep_duration.add_trace(go.Violin(
                x=[stress] * len(subset),
                y=subset["Sleep Duration"],
                name=str(stress),
                fillcolor=colors[i],
                line=dict(color="black", width=1.5),
                opacity=0.9,
                side='both',
                width=0.8,
                box=dict(
                    visible=True,
                    width=0.3,
                    fillcolor='white',
                    line=dict(color='black', width=1.5)
                ),
                meanline=dict(visible=False),
                points=False,
                spanmode='soft',
                bandwidth=0.7
            ))
    
    fig_stress_sleep_duration.update_layout(
        title=dict(
            text="<b>Impact of stress level on sleep duration</b>",
            x=0.5
        ),
        xaxis=dict(
            title=dict(text="Stress Level"),
            categoryorder='array',
            categoryarray=stress_levels,
            tickmode='array',
            tickvals=stress_levels,
            ticktext=[str(level) for level in stress_levels],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)"
        ),
        yaxis=dict(
            title=dict(text="Sleep Duration (hours)"),
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False
        ),
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )

        # Create parallel coordinates plot at the end of the function
    # Create derived features for parallel coordinates
    df_plot = df.copy()
    
        
    # Blood pressure processing
    df_plot["Blood Pressures"] = (
        df_plot["Blood Pressure"].str.split("/").apply(lambda x: (int(x[0]) + int(x[1])) / 2)
    )

    # Create a numerical encoding for Sleep Disorder
    disorder_encoding = {"No Disorder": 0, "Insomnia": 1, "Sleep Apnea": 2}
    df_plot["Sleep Disorder Code"] = df_plot["Sleep Disorder"].map(disorder_encoding)

    # Create a numerical encoding for BMI Category
    bmi_encoding = {"Obese": 0, "Overweight": 1, "Normal": 2}
    df_plot["BMI"] = df_plot["BMI Category"].map(bmi_encoding)

    # Create parallel coordinates plot
    fig_parallel = px.parallel_coordinates(
        df_plot,
        dimensions=[
            "BMI",
            "Physical Activity Level",
            "Daily Steps",
            "Blood Pressures", 
            "Stress Level",
            "Sleep Duration",
            "Quality of Sleep",
        ],
        color="Sleep Disorder Code",
        color_continuous_scale=px.colors.sequential.Agsunset,
        color_continuous_midpoint=1,
        title=""
    )

    # Add annotations
    fig_parallel.update_layout(
        coloraxis_colorbar=dict(
            title="Sleep Disorders",
            tickvals=[0, 1, 2],
            ticktext=["No Disorder", "Insomnia", "Sleep Apnea"],
        ),
        title_x=0.5,
        paper_bgcolor='#faf8ff',
        plot_bgcolor='#faf8ff',
        font=dict(
            color='#E2DFF0' if theme_data and theme_data.get('theme') == 'dark' else '#3D2B61'
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Add the parallel plot to cards at the end

    # Create cards for all charts
    cards = [
        # Full width correlation matrix
        create_graph_card(fig_corr_matrix, className='correlation-graph', is_wide=True),
        
        # Regular cards - first row
        create_graph_card(fig_sleep_age),
        create_graph_card(fig_bmi_sleep),
        
        # Regular cards - second row
        create_graph_card(fig_sleep_hr_kde),
        create_graph_card(fig_occupation_sleep),
        
        # Regular cards - third row
        create_graph_card(fig_disorder_gender),
        create_graph_card(fig_sleep_quality_gender),
        
        # Full width 3D scatter
        create_graph_card(fig_3d_scatter, className='correlation-graph', is_wide=True),
        
        # Regular cards - fourth row
        create_graph_card(fig_activity_disorder),
        create_graph_card(fig_systolic_sleep),
        
        # Regular cards - fifth row
        create_graph_card(fig_stress_sleep_quality),
        create_graph_card(fig_age_disorder),
        
        # Regular cards - sixth row
        create_graph_card(fig_steps_sleep),
        create_graph_card(fig_stress_sleep_duration),

        # Full width parallel coordinates
        create_graph_card(fig_parallel, className='correlation-graph', is_wide=True)
    ]

    
    return html.Div([
    html.H1('Features/Targets Correlations', style={'textAlign': 'center'}),
    html.P('Explore the correlations between features and targets.', 
           className='subtitle', 
           style={'textAlign': 'center'}),
    html.Div(cards, className='graph-container')
])

# Main content callback - simplified
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('theme-store', 'data')]
)
def display_page(pathname, theme_data):
    if pathname == '/distributions':
        return create_distributions_page(theme_data)
    elif pathname == '/correlation-inputs':
        return create_correlation_inputs_page(theme_data)
    elif pathname == '/correlation-targets':
        return create_correlation_targets_page(theme_data)
    elif pathname == '/correlation-all':
        return create_correlation_all_page(theme_data)
    else:
        return create_distributions_page(theme_data)
    
server = app.server

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
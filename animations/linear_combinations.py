import numpy as np
import plotly.graph_objects as go

def show_sum(a, b):

    sum_vec = a + b
    
    fig = go.Figure()

    # Draw vector a
    fig.add_trace(go.Scatter(x=[0, a[0]], y=[0, a[1]], mode='lines+markers',
        line=dict(color='red', width=4), marker=dict(size=10, color='red'), name='a'))
    # Draw vector b
    fig.add_trace(go.Scatter(x=[0, b[0]], y=[0, b[1]], mode='lines+markers',
        line=dict(color='blue', width=4), marker=dict(size=10, color='blue'), name='b'))
    # Draw the sum vector (from origin)
    fig.add_trace(go.Scatter(x=[0, sum_vec[0]], y=[0, sum_vec[1]], mode='lines+markers',
        line=dict(color='green', width=4, dash='dash'), marker=dict(size=10, color='green'), name='a + b'))
    # Draw the triangle (parallelogram) showing the sum visually
    fig.add_trace(go.Scatter(
        x=[a[0], sum_vec[0], b[0], 0],
        y=[a[1], sum_vec[1], b[1], 0],
        mode='lines',
        line=dict(color='gray', width=2, dash='dot'),
        fill='toself',
        fillcolor='rgba(0,200,0,0.1)',
        showlegend=False
    ))

    fig.update_layout(
        title="Sum of Two Vectors",
        xaxis=dict(title='x', range=[-1, max(a[0], b[0], sum_vec[0])+1]),
        yaxis=dict(title='y', range=[-1, max(a[1], b[1], sum_vec[1])+1], scaleanchor='x', scaleratio=1),
        width=600, height=600
    )

    return fig

def show_span(a, b):
    # Range of coefficients for linear combinations
    coeffs = np.linspace(-3, 3, 50)

    # Prepare frames for animation (indexed by alpha and beta)
    frames = []
    for i, alpha in enumerate(coeffs):
        for j, beta in enumerate(coeffs):
            v1 = alpha * a
            v2 = beta * b
            v_sum = v1 + v2
            frame_data = [
                go.Scatter(x=[0, v1[0]], y=[0, v1[1]], mode='lines+markers', line=dict(color='red', width=4), marker=dict(size=8), name='α·a', showlegend=(i==0 and j==0)),
                go.Scatter(x=[v1[0], v_sum[0]], y=[v1[1], v_sum[1]], mode='lines+markers', line=dict(color='blue', width=4), marker=dict(size=8), name='β·b', showlegend=(i==0 and j==0)),
                go.Scatter(x=[0, v_sum[0]], y=[0, v_sum[1]], mode='lines+markers', line=dict(color='green', width=4, dash='dash'), marker=dict(size=10), name='α·a + β·b', showlegend=(i==0 and j==0)),
            ]
            frames.append(go.Frame(data=frame_data, name=f'frame_{i}_{j}', traces=[0,1,2],
                layout=go.Layout(
                    annotations=[
                        dict(
                            x=0.05, y=1.05, xref='paper', yref='paper',
                            text=f'α = {alpha:.2f}, β = {beta:.2f}',
                            showarrow=False, font=dict(size=18, color='black'),
                            align='left', bgcolor='white', bordercolor='black', borderwidth=1
                        )
                    ]
                )
            ))

    # Initial frame (alpha=0, beta=0)
    init_v1 = 0 * a
    init_v2 = 0 * b
    init_sum = init_v1 + init_v2
    data = [
        go.Scatter(x=[0, init_v1[0]], y=[0, init_v1[1]], mode='lines+markers', line=dict(color='red', width=4), marker=dict(size=8), name='α·a'),
        go.Scatter(x=[init_v1[0], init_sum[0]], y=[init_v1[1], init_sum[1]], mode='lines+markers', line=dict(color='blue', width=4), marker=dict(size=8), name='β·b'),
        go.Scatter(x=[0, init_sum[0]], y=[0, init_sum[1]], mode='lines+markers', line=dict(color='green', width=4, dash='dash'), marker=dict(size=10), name='α·a + β·b'),
    ]

    layout = go.Layout(
        title="Linear Combinations of Two Vectors (Two Sliders)",
        width=800, height=800,
        margin=dict(l=40, r=40, t=80, b=120), # Add extra space at the bottom for sliders
        xaxis=dict(range=[-10, 10], zeroline=True, zerolinecolor='black'),
        yaxis=dict(range=[-10, 10], zeroline=True, zerolinecolor='black', scaleanchor='x', scaleratio=1),
        annotations=[
            dict(
                x=0.05, y=1.05, xref='paper', yref='paper',
                text=f'α = {0:.2f}, β = {0:.2f}',
                showarrow=False, font=dict(size=18, color='black'),
                align='left', bgcolor='white', bordercolor='black', borderwidth=1
            )
        ],
        updatemenus=[] # No play/pause buttons for manual slider control
    )

    # Create two sliders: one for alpha, one for beta, both placed below the plot
    alpha_steps = []
    for i, alpha in enumerate(coeffs):
        alpha_steps.append(dict(
            method='animate',
            args=[[f'frame_{i}_0'],
                {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
            label=f'α={alpha:.2f}'
        ))

    beta_steps = []
    for j, beta in enumerate(coeffs):
        beta_steps.append(dict(
            method='animate',
            args=[[f'frame_0_{j}'],
                {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
            label=f'β={beta:.2f}'
        ))

    sliders = [
        dict(
            active=0,
            steps=alpha_steps,
            currentvalue={"prefix": "α: "},
            pad={"t": 0, "b": 0},
            x=0.1, y=-0.18, xanchor='left', yanchor='top', # Place below plot
            len=0.8,
        ),
        dict(
            active=0,
            steps=beta_steps,
            currentvalue={"prefix": "β: "},
            pad={"t": 40, "b": 0},
            x=0.1, y=-0.28, xanchor='left', yanchor='top', # Place further below plot
            len=0.8,
        )
    ]

    layout.sliders = sliders

    fig = go.Figure(data=data, layout=layout, frames=frames)    
    return fig

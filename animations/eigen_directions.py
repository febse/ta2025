import numpy as np
import plotly.graph_objects as go


def show_eigen_directions(A, vecs):


    # Compute eigenvectors and eigenvalues of A
    eigvals, eigvecs = np.linalg.eig(A)
    v1 = eigvecs[:, 0]
    v2 = eigvecs[:, 1]

    # Grid parameters
    grid_range = np.arange(-5, 6, 1)
    eig_grid_range = np.arange(-4, 5, 1)
    n_frames = 30

    # Standard basis vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])

    # Helper to interpolate between identity and A
    def interpolate_matrix(t):
        return (1-t)*np.eye(2) + t*A

    # Helper to interpolate eigenvectors
    def interpolate_eigenvectors(t):
        v1_t = (1-t)*e1 + t*v1
        v2_t = (1-t)*e2 + t*v2
        return v1_t, v2_t

    # Helper to draw grid lines in interpolated eigenvector basis
    def moving_eig_grid_lines(v1_t, v2_t):
        lines = []
        for a in eig_grid_range:
            start = a * v1_t + eig_grid_range[0] * v2_t
            end = a * v1_t + eig_grid_range[-1] * v2_t
            lines.append(go.Scatter(x=[start[0], end[0]], y=[start[1], end[1]], mode='lines', line=dict(color='red', width=1), showlegend=False, hoverinfo='skip'))
        for b in eig_grid_range:
            start = b * v2_t + eig_grid_range[0] * v1_t
            end = b * v2_t + eig_grid_range[-1] * v1_t
            lines.append(go.Scatter(x=[start[0], end[0]], y=[start[1], end[1]], mode='lines', line=dict(color='blue', width=1), showlegend=False, hoverinfo='skip'))
        return lines

    # Prepare frames for animation
    frames = []
    for k in range(n_frames+1):
        t = k / n_frames
        T = interpolate_matrix(t)
        v1_t, v2_t = interpolate_eigenvectors(t)
        lines = []
        # Draw original grid (static, gray)
        for x in grid_range:
            lines.append(go.Scatter(x=[x, x], y=[grid_range[0], grid_range[-1]], mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
        for y in grid_range:
            lines.append(go.Scatter(x=[grid_range[0], grid_range[-1]], y=[y, y], mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
        # Draw moving eigenvector grid (red/blue, rotates)
        lines.extend(moving_eig_grid_lines(v1_t, v2_t))
        # Draw the vectors, transformed by T
        colors = ['magenta', 'cyan', 'orange', 'brown', 'green']
        for i, v in enumerate(vecs):
            tip = T @ v
            lines.append(go.Scatter(x=[0, tip[0]], y=[0, tip[1]], mode='lines+markers', line=dict(color=colors[i], width=4), marker=dict(size=10), name=f'vec {i+1}: {tuple(np.round(v,2))}'))
        # Draw eigenvectors from origin (purple/orange, fixed, not normalized)
        lines.append(go.Scatter(x=[0, v1[0]], y=[0, v1[1]], mode='lines+markers', line=dict(color='purple', width=4), marker=dict(size=10), name='eigvec 1'))
        lines.append(go.Scatter(x=[0, v2[0]], y=[0, v2[1]], mode='lines+markers', line=dict(color='orange', width=4), marker=dict(size=10), name='eigvec 2'))
        # Add annotation for t
        ann = [dict(x=0.05, y=1.05, xref='paper', yref='paper', text=f't = {t:.2f}', showarrow=False, font=dict(size=18, color='black'), align='left', bgcolor='white', bordercolor='black', borderwidth=1)]
        frames.append(go.Frame(data=lines, name=f'frame_{k}', layout=go.Layout(annotations=ann)))

    # Initial frame (t=0)
    T0 = interpolate_matrix(0)
    v1_0, v2_0 = interpolate_eigenvectors(0)
    lines0 = []
    for x in grid_range:
        lines0.append(go.Scatter(x=[x, x], y=[grid_range[0], grid_range[-1]], mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
    for y in grid_range:
        lines0.append(go.Scatter(x=[grid_range[0], grid_range[-1]], y=[y, y], mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
    lines0.extend(moving_eig_grid_lines(v1_0, v2_0))
    colors = ['magenta', 'cyan', 'orange', 'brown']
    for i, v in enumerate(vecs):
        tip = T0 @ v
        lines0.append(go.Scatter(x=[0, tip[0]], y=[0, tip[1]], mode='lines+markers', line=dict(color=colors[i], width=4), marker=dict(size=10), name=f'vec {i+1}: {tuple(np.round(v,2))}'))
    lines0.append(go.Scatter(x=[0, v1[0]], y=[0, v1[1]], mode='lines+markers', line=dict(color='purple', width=4), marker=dict(size=10), name='eigvec 1'))
    lines0.append(go.Scatter(x=[0, v2[0]], y=[0, v2[1]], mode='lines+markers', line=dict(color='orange', width=4), marker=dict(size=10), name='eigvec 2'))

    layout = go.Layout(
        title="Transformation Animation: Rotating Grid and Vectors Toward Eigenvectors",
        width=900, height=900,
        xaxis=dict(range=[-8, 8], zeroline=True, zerolinecolor='black', showgrid=False),
        yaxis=dict(range=[-8, 8], zeroline=True, zerolinecolor='black', showgrid=False, scaleanchor='x', scaleratio=1),
        legend=dict(x=0.7, y=0.95),
        margin=dict(l=40, r=40, t=80, b=40),
        updatemenus=[
            dict(type='buttons', showactive=False, y=1.08, x=1.15, xanchor='right', yanchor='top',
                buttons=[
                    dict(label='Play', method='animate', args=[[f'frame_{k}' for k in range(n_frames+1)], {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label='Pause', method='animate', args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
                ])
        ],
        sliders=[
            dict(
                active=0,
                steps=[dict(method='animate', args=[[f'frame_{k}'], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}], label=f't={k/n_frames:.2f}') for k in range(n_frames+1)],
                x=0.1, y=-0.08, xanchor='left', yanchor='top', len=0.8,
                currentvalue={"prefix": "t: "},
                pad={"t": 40, "b": 0}
            )
        ]
    )

    fig = go.Figure(data=lines0, layout=layout, frames=frames)    
    return fig

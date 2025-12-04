import numpy as np
import plotly.graph_objects as go

def show_dot_product_cosine():
    # Parameters for animation
    n_frames = 60
    angles = np.linspace(0, 2 * np.pi, n_frames)
    scales = np.linspace(0.2, 2.0, 10)

    # Initial values
    init_angle = np.pi / 4
    init_scale1 = 1.0
    init_scale2 = 1.0

    def get_vectors(angle, scale1, scale2):
        # v1 is always at 0 degrees
        v1 = scale1 * np.array([np.cos(0), np.sin(0)])
        # v2 is at 'angle'
        v2 = scale2 * np.array([np.cos(angle), np.sin(angle)])
        return v1, v2

    def get_cosine_and_dot(v1, v2):
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        cosine = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        return cosine, dot

    # Create frames for animation
    frames = []
    for i, angle in enumerate(angles):
        v1, v2 = get_vectors(angle, init_scale1, init_scale2)
        cosine, dot = get_cosine_and_dot(v1, v2)
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=[0, v1[0]], y=[0, v1[1]],
                    mode='lines+markers', line=dict(color='steelblue', width=4),
                    marker=dict(size=12, color='steelblue'),
                    name='v1'
                ),
                go.Scatter(
                    x=[0, v2[0]], y=[0, v2[1]],
                    mode='lines+markers', line=dict(color='firebrick', width=4),
                    marker=dict(size=12, color='firebrick'),
                    name='v2'
                ),
                go.Scatter(
                    x=[v1[0], v2[0]], y=[v1[1], v2[1]],
                    mode='markers', marker=dict(size=0), showlegend=False
                )
            ],
            layout=go.Layout(
                annotations=[
                    dict(
                        x=0.98, y=0.02, xref='paper', yref='paper',
                        text=f'Angle: {np.degrees(angle):.1f}°<br>Cosine: {cosine:.3f}<br>Dot: {dot:.3f}',
                        showarrow=False, font=dict(size=18), align='left', xanchor='right', yanchor='bottom'
                    )
                ]
            ),
            name=f'frame_{i}'
        )
        frames.append(frame)

    # Initial vectors
    v1, v2 = get_vectors(init_angle, init_scale1, init_scale2)
    cosine, dot = get_cosine_and_dot(v1, v2)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=[0, v1[0]], y=[0, v1[1]],
                mode='lines+markers', line=dict(color='steelblue', width=4),
                marker=dict(size=12, color='steelblue'),
                name='v1'
            ),
            go.Scatter(
                x=[0, v2[0]], y=[0, v2[1]],
                mode='lines+markers', line=dict(color='firebrick', width=4),
                marker=dict(size=12, color='firebrick'),
                name='v2'
            )
        ],
        layout=go.Layout(
            xaxis=dict(range=[-2.2, 2.2], zeroline=True, scaleanchor='y', scaleratio=1),
            yaxis=dict(range=[-2.2, 2.2], zeroline=True),
            width=600, height=600,
            title="Two Vectors on the Unit Circle",
            annotations=[
                dict(
                    x=0.98, y=0.02, xref='paper', yref='paper',
                    text=f'Angle: {np.degrees(init_angle):.1f}°<br>Cosine: {cosine:.3f}<br>Dot: {dot:.3f}',
                    showarrow=False, font=dict(size=18), align='left', xanchor='right', yanchor='bottom'
                )
            ],
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(label='Play', method='animate', args=[None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True}]),
                        dict(label='Pause', method='animate', args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                    ],
                    x=0.1, y=1.1, xanchor='left', yanchor='top'
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(method='animate', args=[[f'frame_{i}'], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], label=f'{np.degrees(angle):.1f}°')
                        for i, angle in enumerate(angles)
                    ],
                    active=int(init_angle / (2 * np.pi) * n_frames),
                    x=0.1, y=0, xanchor='left', yanchor='top',
                    currentvalue=dict(prefix='Angle: ', font=dict(size=16))
                )
            ]
        ),
        frames=frames
    )

    return fig
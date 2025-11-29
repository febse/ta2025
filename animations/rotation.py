import numpy as np
import plotly.graph_objects as go

# Parameters
L = 5                      # grid extends from -L to L
angle_degs = np.arange(0, 361, 15)  # rotation angles (degrees)
np.random.seed(42)

# Rotation matrix
def R(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

# Heart mask: implicit heart equation
# Inside if (x^2 + y^2 - 1)^3 - x^2*y^3 <= 0
# Slightly scale coordinates to fit heart snugly in the grid
HEART_SCALE = 1.2

def in_heart(x, y):
    xs = x / HEART_SCALE
    ys = y / HEART_SCALE
    val = (xs*xs + ys*ys - 1)**3 - (xs*xs)*(ys**3)
    return val <= 0

# Decide line color based on fraction of sampled points inside heart
# Returns a hex color string

def heart_line_color(XY_samples):
    inside_count = sum(in_heart(x, y) for x, y in XY_samples)
    frac = inside_count / len(XY_samples)
    if frac >= 0.66:
        return '#D0162F'  # deep red
    elif frac >= 0.33:
        return '#F76C8E'  # pink
    else:
        return 'lightgray'

# Build unit grid lines (unrotated endpoints)
vert_lines = [((x, -L), (x, L)) for x in range(-L, L+1)]
horz_lines = [((-L, y), (L, y)) for y in range(-L, L+1)]

# Basis vectors (unrotated)
e1 = np.array([1.0, 0.0])
e2 = np.array([0.0, 1.0])

# Heart parametric curve for points (outline)
# Classic heart curve: x = 16 sin^3 t, y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t
# Scale to fit within [-L+0.5, L-0.5]

def heart_param_points(n=40):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = 16*np.sin(t)**3
    y = (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t))
    # Normalize to [-1,1] range approximately
    x /= 18.0
    y /= 18.0
    # Scale to grid
    scale = (L - 0.8)  # keep some margin
    return np.column_stack([x*scale, y*scale])

n_pts = 40
pts = heart_param_points(n=n_pts)

# Helper: rotate a list of 2D points
def rotate_points(P, theta):
    rot = R(theta)
    return (rot @ P.T).T

# Helper: sample N points along a segment defined by endpoints P0, P1
def sample_segment(P0, P1, n=11):
    ts = np.linspace(0.0, 1.0, n)
    return [(P0[0]*(1-t) + P1[0]*t, P0[1]*(1-t) + P1[1]*t) for t in ts]

# Create initial figure data (theta = 0)
init_theta = np.deg2rad(0)


def show_rotation():
    # Original grid (fixed, low opacity)
    orig_grid_traces = []
    for (x1, y1), (x2, y2) in vert_lines + horz_lines:
        P = np.array([[x1, y1], [x2, y2]])
        orig_grid_traces.append(go.Scatter(
            x=P[:, 0], y=P[:, 1], mode='lines',
            line=dict(color='lightgray', width=1),
            opacity=0.25,
            hoverinfo='skip', showlegend=False
        ))

    # Rotated grid traces (vertical then horizontal), color by heart
    grid_traces = []
    for (x1, y1), (x2, y2) in vert_lines + horz_lines:
        P0, P1 = np.array([x1, y1]), np.array([x2, y2])
        Pr = rotate_points(np.vstack([P0, P1]), init_theta)
        samples = sample_segment(Pr[0], Pr[1], n=21)
        col = heart_line_color(samples)
        grid_traces.append(go.Scatter(
            x=[Pr[0, 0], Pr[1, 0]], y=[Pr[0, 1], Pr[1, 1]], mode='lines',
            line=dict(color=col, width=1.5),
            hoverinfo='skip', showlegend=False
        ))

    # Basis vectors
    e1r = rotate_points(np.array([e1]), init_theta)[0]
    e2r = rotate_points(np.array([e2]), init_theta)[0]
    vec_e1 = go.Scatter(x=[0, e1r[0]], y=[0, e1r[1]], mode='lines+markers',
                        line=dict(color='#1f77b4', width=3), marker=dict(size=6),
                        name='e1')
    vec_e2 = go.Scatter(x=[0, e2r[0]], y=[0, e2r[1]], mode='lines+markers',
                        line=dict(color='#2ca02c', width=3), marker=dict(size=6),
                        name='e2')

    # Points: original heart outline and rotated
    pts_rot0 = rotate_points(pts, init_theta)
    pts_orig_trace = go.Scatter(x=pts[:, 0], y=pts[:, 1], mode='markers',
                                marker=dict(color='orange', size=7, symbol='circle-open'),
                                name='Heart pts')
    rot_pt_colors = ['#D0162F' if in_heart(x, y) else 'firebrick' for x, y in pts_rot0]
    pts_rot_trace = go.Scatter(x=pts_rot0[:, 0], y=pts_rot0[:, 1], mode='markers',
                            marker=dict(color=rot_pt_colors, size=8),
                            name='Rotated pts')

    # Matrix annotation (initial)
    matrix_R = R(init_theta)
    matrix_text = f"R(θ=0°) = [[{matrix_R[0,0]:.3f}, {matrix_R[0,1]:.3f}],<br>          [{matrix_R[1,0]:.3f}, {matrix_R[1,1]:.3f}]]"
    matrix_annotation = dict(
        x=0.02, y=0.98, xref='paper', yref='paper',
        text=matrix_text, showarrow=False,
        font=dict(family='monospace', size=12, color='black'),
        bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1,
        xanchor='left', yanchor='top'
    )

    # Assemble initial data order: original grid (fixed), rotated grid, basis vectors, original points, rotated points
    data = orig_grid_traces + grid_traces + [vec_e1, vec_e2, pts_orig_trace, pts_rot_trace]

    # Build frames for each angle
    frames = []
    for deg in angle_degs:
        theta = np.deg2rad(deg)
        frame_data = []

        # Original grid (stays fixed)
        for (x1, y1), (x2, y2) in vert_lines + horz_lines:
            P = np.array([[x1, y1], [x2, y2]])
            frame_data.append(go.Scatter(x=P[:, 0], y=P[:, 1], mode='lines',
                                        line=dict(color='lightgray', width=1),
                                        opacity=0.25,
                                        hoverinfo='skip', showlegend=False))

        # Update rotated grid lines with heart-based colors
        for (x1, y1), (x2, y2) in vert_lines + horz_lines:
            P0, P1 = np.array([x1, y1]), np.array([x2, y2])
            Pr = rotate_points(np.vstack([P0, P1]), theta)
            samples = sample_segment(Pr[0], Pr[1], n=21)
            col = heart_line_color(samples)
            frame_data.append(go.Scatter(x=[Pr[0, 0], Pr[1, 0]], y=[Pr[0, 1], Pr[1, 1]], mode='lines',
                                        line=dict(color=col, width=1.5),
                                        hoverinfo='skip', showlegend=False))

        # Update basis vectors
        e1r = rotate_points(np.array([e1]), theta)[0]
        e2r = rotate_points(np.array([e2]), theta)[0]
        frame_data.append(go.Scatter(x=[0, e1r[0]], y=[0, e1r[1]], mode='lines+markers',
                                    line=dict(color='#1f77b4', width=3), marker=dict(size=6),
                                    name='e1'))
        frame_data.append(go.Scatter(x=[0, e2r[0]], y=[0, e2r[1]], mode='lines+markers',
                                    line=dict(color='#2ca02c', width=3), marker=dict(size=6),
                                    name='e2'))

        # Original heart points (stay fixed)
        frame_data.append(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode='markers',
                                    marker=dict(color='orange', size=7, symbol='circle-open'),
                                    name='Heart pts'))
        # Rotated points, recolor by heart membership
        pts_rot = rotate_points(pts, theta)
        rot_pt_colors = ['#D0162F' if in_heart(x, y) else 'firebrick' for x, y in pts_rot]
        frame_data.append(go.Scatter(x=pts_rot[:, 0], y=pts_rot[:, 1], mode='markers',
                                    marker=dict(color=rot_pt_colors, size=8),
                                    name='Rotated pts'))

        # Update matrix annotation
        matrix_R = R(theta)
        matrix_text = f"R(θ={deg}°) = [[{matrix_R[0,0]:.3f}, {matrix_R[0,1]:.3f}],<br>              [{matrix_R[1,0]:.3f}, {matrix_R[1,1]:.3f}]]"
        frame_annotation = dict(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=matrix_text, showarrow=False,
            font=dict(family='monospace', size=12, color='black'),
            bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1,
            xanchor='left', yanchor='top'
        )

        frames.append(go.Frame(data=frame_data, name=f"deg_{deg}", layout={'annotations': [frame_annotation]}))

    # Build slider steps for degrees
    slider_steps = []
    for deg in angle_degs:
        step = {
            'label': f'{deg}°',
            'method': 'animate',
            'args': [[f'deg_{deg}'], {
                'mode': 'immediate',
                'frame': {'duration': 0, 'redraw': True},
                'transition': {'duration': 0}
            }]
        }
        slider_steps.append(step)

    # Figure and layout
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        title='Rotating Grid Colored as Heart (Rotation Matrix)',
        xaxis=dict(range=[-L-1, L+1], zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(range=[-L-1, L+1], zeroline=True, zerolinewidth=2, zerolinecolor='black',
                scaleanchor='x', scaleratio=1),
        width=750, height=750,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
        annotations=[matrix_annotation],
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None], {'mode': 'immediate'}]}
            ]
        }],
        sliders=[{
            'active': 0,
            'currentvalue': {'prefix': 'θ: ', 'suffix': '', 'visible': True},
            'pad': {'t': 40},
            'steps': slider_steps
        }]
    )

    return fig

import numpy as np
import plotly.graph_objects as go

# Parameters
L = 5                      # grid extends from -L to L
angle_degs = np.arange(0, 361, 15)  # rotation angles (degrees)
np.random.seed(42)

# Rotation matrix (2D)

def R_2D(theta):
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

# Helper: apply a sequence of transformation matrices to points
def apply_transformation(points, matrices):
    """Apply a sequence of 2D transformation matrices to points.
    
    Args:
        points: Nx2 array of 2D points
        matrices: list of 2x2 transformation matrices (applied in order)
    
    Returns:
        Transformed Nx2 array of points
    """
    result = points.copy()
    for M in matrices:
        result = (M @ result.T).T
    return result

# Helper: compute rotation angle from transformed basis vectors
def compute_rotation_angle(e1_transformed, e2_transformed):
    """Compute the rotation angle in degrees from transformed basis vectors.
    
    Args:
        e1_transformed: transformed e1 basis vector [x, y]
        e2_transformed: transformed e2 basis vector [x, y]
    
    Returns:
        Rotation angle in degrees
    """
    
    # Use atan2 on the transformed e1 vector
    angle_rad = np.arctan2(e1_transformed[1], e1_transformed[0])
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg % 360

# Helper: sample N points along a segment defined by endpoints P0, P1
def sample_segment(P0, P1, n=11):
    ts = np.linspace(0.0, 1.0, n)
    return [(P0[0]*(1-t) + P1[0]*t, P0[1]*(1-t) + P1[1]*t) for t in ts]

# Create initial figure data (theta = 0)
init_theta = np.deg2rad(0)


def show_transformation(matrices_sequence, points, L=5, 
                        title='Transformation Animation', color_func=None):
    """Show animated transformation of grid and points.
    
    Args:
        matrices_sequence: list of lists of 2x2 transformation matrices, one per frame
        points: Nx2 array of points to transform
        L: grid extent for plot range
        title: plot title
        color_func: optional function(samples) -> color for grid lines
    
    Returns:
        Plotly Figure object
    """
    # Build unit grid lines (unrotated endpoints)
    vert_lines = [((x, -L), (x, L)) for x in range(-L, L+1)]
    horz_lines = [((-L, y), (L, y)) for y in range(-L, L+1)]
    grid_lines = vert_lines + horz_lines
    
    # Basis vectors (unrotated)
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    
    # Original grid (fixed, low opacity)
    orig_grid_traces = []
    for (x1, y1), (x2, y2) in grid_lines:
        P = np.array([[x1, y1], [x2, y2]])
        orig_grid_traces.append(go.Scatter(
            x=P[:, 0], y=P[:, 1], mode='lines',
            line=dict(color='lightgray', width=1),
            opacity=0.25,
            hoverinfo='skip', showlegend=False
        ))

    # Original grid (fixed, low opacity)
    orig_grid_traces = []
    for (x1, y1), (x2, y2) in grid_lines:
        P = np.array([[x1, y1], [x2, y2]])
        orig_grid_traces.append(go.Scatter(
            x=P[:, 0], y=P[:, 1], mode='lines',
            line=dict(color='lightgray', width=1),
            opacity=0.25,
            hoverinfo='skip', showlegend=False
        ))

    # Initial transformation (first frame - identity or first matrix set)
    initial_matrices = matrices_sequence[0] if matrices_sequence else []
    
    # Transformed grid traces (vertical then horizontal), color by optional func
    grid_traces = []
    for (x1, y1), (x2, y2) in grid_lines:
        P0, P1 = np.array([x1, y1]), np.array([x2, y2])
        Pr = apply_transformation(np.vstack([P0, P1]), initial_matrices)
        
        if color_func:
            samples = sample_segment(Pr[0], Pr[1], n=21)
            col = color_func(samples)
        else:
            col = 'gray'
            
        grid_traces.append(go.Scatter(
            x=[Pr[0, 0], Pr[1, 0]], y=[Pr[0, 1], Pr[1, 1]], mode='lines',
            line=dict(color=col, width=1.5),
            hoverinfo='skip', showlegend=False
        ))

    # Basis vectors
    e1_transformed = apply_transformation(np.array([e1]), initial_matrices)[0]
    e2_transformed = apply_transformation(np.array([e2]), initial_matrices)[0]
    vec_e1 = go.Scatter(x=[0, e1_transformed[0]], y=[0, e1_transformed[1]], mode='lines+markers',
                        line=dict(color='#1f77b4', width=3), marker=dict(size=6),
                        name='e1')
    vec_e2 = go.Scatter(x=[0, e2_transformed[0]], y=[0, e2_transformed[1]], mode='lines+markers',
                        line=dict(color='#2ca02c', width=3), marker=dict(size=6),
                        name='e2')

    # Points: original and transformed
    pts_transformed = apply_transformation(points, initial_matrices)
    pts_orig_trace = go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers',
                                marker=dict(color='orange', size=7, symbol='circle-open'),
                                name='Original pts')
    
    if color_func:
        pt_colors = ['#D0162F' if in_heart(x, y) else 'firebrick' for x, y in pts_transformed]
    else:
        pt_colors = 'firebrick'
    
    pts_transformed_trace = go.Scatter(x=pts_transformed[:, 0], y=pts_transformed[:, 1], mode='markers',
                                       marker=dict(color=pt_colors, size=8),
                                       name='Transformed pts')

    # Compute rotation angle from basis vectors
    rotation_deg = compute_rotation_angle(e1_transformed, e2_transformed)
    
    # Compute combined transformation matrix for annotation
    combined_matrix = np.eye(2)
    for M in initial_matrices:
        combined_matrix = M @ combined_matrix
    
    matrix_text = f"θ={rotation_deg:.1f}° | T = [[{combined_matrix[0,0]:.3f}, {combined_matrix[0,1]:.3f}],<br>              [{combined_matrix[1,0]:.3f}, {combined_matrix[1,1]:.3f}]]"
    matrix_annotation = dict(
        x=0.02, y=0.98, xref='paper', yref='paper',
        text=matrix_text, showarrow=False,
        font=dict(family='monospace', size=12, color='black'),
        bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1,
        xanchor='left', yanchor='top'
    )

    # Assemble initial data order: original grid, transformed grid, basis vectors, original points, transformed points
    data = orig_grid_traces + grid_traces + [vec_e1, vec_e2, pts_orig_trace, pts_transformed_trace]

    # Build frames for each transformation
    frames = []
    for frame_idx, matrices in enumerate(matrices_sequence):
        frame_data = []

        # Original grid (stays fixed)
        for (x1, y1), (x2, y2) in grid_lines:
            P = np.array([[x1, y1], [x2, y2]])
            frame_data.append(go.Scatter(x=P[:, 0], y=P[:, 1], mode='lines',
                                        line=dict(color='lightgray', width=1),
                                        opacity=0.25,
                                        hoverinfo='skip', showlegend=False))

        # Update transformed grid lines
        for (x1, y1), (x2, y2) in grid_lines:
            P0, P1 = np.array([x1, y1]), np.array([x2, y2])
            Pr = apply_transformation(np.vstack([P0, P1]), matrices)
            
            if color_func:
                samples = sample_segment(Pr[0], Pr[1], n=21)
                col = color_func(samples)
            else:
                col = 'gray'
                
            frame_data.append(go.Scatter(x=[Pr[0, 0], Pr[1, 0]], y=[Pr[0, 1], Pr[1, 1]], mode='lines',
                                        line=dict(color=col, width=1.5),
                                        hoverinfo='skip', showlegend=False))

        # Update basis vectors
        e1_t = apply_transformation(np.array([e1]), matrices)[0]
        e2_t = apply_transformation(np.array([e2]), matrices)[0]
        frame_data.append(go.Scatter(x=[0, e1_t[0]], y=[0, e1_t[1]], mode='lines+markers',
                                    line=dict(color='#1f77b4', width=3), marker=dict(size=6),
                                    name='e1'))
        frame_data.append(go.Scatter(x=[0, e2_t[0]], y=[0, e2_t[1]], mode='lines+markers',
                                    line=dict(color='#2ca02c', width=3), marker=dict(size=6),
                                    name='e2'))

        # Original points (stay fixed)
        frame_data.append(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers',
                                    marker=dict(color='orange', size=7, symbol='circle-open'),
                                    name='Original pts'))
        
        # Transformed points
        pts_t = apply_transformation(points, matrices)
        if color_func:
            pt_colors = ['#D0162F' if in_heart(x, y) else 'firebrick' for x, y in pts_t]
        else:
            pt_colors = 'firebrick'
        frame_data.append(go.Scatter(x=pts_t[:, 0], y=pts_t[:, 1], mode='markers',
                                    marker=dict(color=pt_colors, size=8),
                                    name='Transformed pts'))

        # Update matrix annotation
        rotation_deg = compute_rotation_angle(e1_t, e2_t)
        combined_matrix = np.eye(2)
        for M in matrices:
            combined_matrix = M @ combined_matrix
            
        matrix_text = f"θ={rotation_deg:.1f}° | T = [[{combined_matrix[0,0]:.3f}, {combined_matrix[0,1]:.3f}],<br>                  [{combined_matrix[1,0]:.3f}, {combined_matrix[1,1]:.3f}]]"
        frame_annotation = dict(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=matrix_text, showarrow=False,
            font=dict(family='monospace', size=12, color='black'),
            bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1,
            xanchor='left', yanchor='top'
        )

        frames.append(go.Frame(data=frame_data, name=f"frame_{frame_idx}", 
                              layout={'annotations': [frame_annotation]}))

    # Build slider steps
    slider_steps = []
    for frame_idx in range(len(matrices_sequence)):
        rotation_deg = compute_rotation_angle(
            apply_transformation(np.array([e1]), matrices_sequence[frame_idx])[0],
            apply_transformation(np.array([e2]), matrices_sequence[frame_idx])[0]
        )
        step = {
            'label': f'{rotation_deg:.0f}°',
            'method': 'animate',
            'args': [[f'frame_{frame_idx}'], {
                'mode': 'immediate',
                'frame': {'duration': 0, 'redraw': True},
                'transition': {'duration': 0}
            }]
        }
        slider_steps.append(step)

    # Figure and layout
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        title=title,
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


def generate_rotation_matrices(angle_degs):
    """Generate rotation matrices for a sequence of angles.
    
    Args:
        angle_degs: array or list of rotation angles in degrees
    
    Returns:
        List of matrix lists, where each element is [R(theta)] for that angle
    """
    matrices_sequence = []
    for deg in angle_degs:
        theta = np.deg2rad(deg)
        matrices_sequence.append([R_2D(theta)])
    return matrices_sequence


def show_rotation(angle_degs=None, L=5):
    """Show rotation transformation animation with heart-colored grid.
    
    Args:
        angle_degs: array of rotation angles in degrees (default: 0 to 360 in 15° steps)
        L: grid extent
    
    Returns:
        Plotly Figure object
    """
    if angle_degs is None:
        angle_degs = np.arange(0, 361, 15)
    
    # Heart parametric curve for points (outline)
    pts = heart_param_points(n=20)
    
    # Generate rotation matrices
    matrices_sequence = generate_rotation_matrices(angle_degs)
    
    # Show transformation with heart coloring
    return show_transformation(
        matrices_sequence=matrices_sequence,
        points=pts,
        L=L,
        title='Rotating Grid Colored as Heart (Rotation Matrix)',
        color_func=heart_line_color
    )


if __name__ == '__main__':
    fig = show_rotation()
    fig.show()

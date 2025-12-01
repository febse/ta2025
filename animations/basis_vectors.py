import numpy as np
import plotly.graph_objects as go

def show_basis_vectors(u, v):
    # Create a grid in the original basis (standard grid)
    grid_range = np.arange(-5, 6, 1)
    lines = []
    for x in grid_range:
        # Vertical lines (x fixed, y varies)
        lines.append(go.Scatter(x=[x, x], y=[grid_range[0], grid_range[-1]], mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
    for y in grid_range:
        # Horizontal lines (y fixed, x varies)
        lines.append(go.Scatter(x=[grid_range[0], grid_range[-1]], y=[y, y], mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))

    # Create a grid in the new basis (u, v)
    basis_grid_range = np.arange(-4, 5, 1)
    for a in basis_grid_range:
        # Lines parallel to v (move along u)
        start = a * u + basis_grid_range[0] * v
        end = a * u + basis_grid_range[-1] * v
        lines.append(go.Scatter(x=[start[0], end[0]], y=[start[1], end[1]], mode='lines', line=dict(color='red', width=1), showlegend=False, hoverinfo='skip'))
    for b in basis_grid_range:
        # Lines parallel to u (move along v)
        start = b * v + basis_grid_range[0] * u
        end = b * v + basis_grid_range[-1] * u
        lines.append(go.Scatter(x=[start[0], end[0]], y=[start[1], end[1]], mode='lines', line=dict(color='blue', width=1), showlegend=False, hoverinfo='skip'))

    # Draw the basis vectors from the origin
    lines.append(go.Scatter(x=[0, u[0]], y=[0, u[1]], mode='lines+markers', line=dict(color='red', width=4), marker=dict(size=10), name='u = [1, 1]'))
    lines.append(go.Scatter(x=[0, v[0]], y=[0, v[1]], mode='lines+markers', line=dict(color='blue', width=4), marker=dict(size=10), name='v = [-2, 2]'))

    # Set up the layout
    layout = go.Layout(
        title="Original Grid and Grid Relative to New Basis",
        width=700, height=700,
        xaxis=dict(range=[-6, 6], zeroline=True, zerolinecolor='black', showgrid=False),
        yaxis=dict(range=[-6, 6], zeroline=True, zerolinecolor='black', showgrid=False, scaleanchor='x', scaleratio=1),
        legend=dict(x=0.7, y=0.95),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    # Add a vector w = (4, 1) in standard basis and show its coordinates in the (u, v) basis
    w = np.array([4, 1])

    # Compute the change of basis matrix from (u, v) to standard basis
    # The columns of the matrix are u and v
    B = np.column_stack((u, v))
    # To get coordinates in the (u, v) basis, solve B @ [a, b] = w
    coords = np.linalg.solve(B, w)
    # coords[0] is the coefficient for u, coords[1] for v

    # Plot the vector w in standard basis
    lines.append(go.Scatter(x=[0, w[0]], y=[0, w[1]], mode='lines+markers', line=dict(color='orange', width=4, dash='dash'), marker=dict(size=12), name='w = (4, 1)'))

    # Annotate the coordinates in the (u, v) basis
    lines.append(go.Scatter(x=[w[0]], y=[w[1]], mode='markers+text', marker=dict(size=1, color='rgba(0,0,0,0)'),
        text=[f"({coords[0]:.2f})·u + ({coords[1]:.2f})·v"], textposition='top right', showlegend=False))

    # Create the figure
    fig = go.Figure(data=lines, layout=layout)
    return fig

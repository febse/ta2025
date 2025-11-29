from manim import *

class SVDRotation(Scene):
    def construct(self):
        # Set up axes
        axes = Axes(
            x_range=[48, 64, 2],
            y_range=[48, 64, 2],
            x_length=7,
            y_length=7,
            axis_config={"include_numbers": True, "font_size": 24},
        )
        
        # Labels
        x_label = axes.get_x_axis_label("Left hand", direction=DOWN)
        y_label = axes.get_y_axis_label("Right hand", direction=UP)
        
        # Generate the data (same as above)
        np.random.seed(627)
        n = 15
        x_data = np.random.uniform(50, 62, size=n)
        y_data = x_data + np.random.normal(0, 0.5, n)
        
        # Center the data
        x_mean = x_data.mean()
        y_mean = y_data.mean()
        x_centered = x_data - x_mean
        y_centered = y_data - y_mean
        
        # # Create data matrix
        # XY = np.vstack([x_centered, y_centered]).T
        
        # # Perform SVD
        # U, S, VT = np.linalg.svd(XY, full_matrices=False)
        
        # # Create dots for data points
        # dots = VGroup(*[
        #     Dot(axes.c2p(x_data[i], y_data[i]), color=BLUE, radius=0.08)
        #     for i in range(n)
        # ])
        
        # # Create the original axes lines through the mean
        # origin = axes.c2p(x_mean, y_mean)
        
        # # Original axis directions (unit vectors)
        # x_axis_line = Arrow(
        #     axes.c2p(x_mean - 5, y_mean),
        #     axes.c2p(x_mean + 5, y_mean),
        #     color=GREEN,
        #     buff=0,
        #     stroke_width=3
        # )
        # y_axis_line = Arrow(
        #     axes.c2p(x_mean, y_mean - 5),
        #     axes.c2p(x_mean, y_mean + 5),
        #     color=GREEN,
        #     buff=0,
        #     stroke_width=3
        # )
        
        # # First singular vector direction (in original space)
        # # VT[0] gives us the direction in the centered coordinate system
        # v1 = VT[0]  # First right singular vector
        
        # # Scale for visualization
        # scale = 6
        # sv1_line = Arrow(
        #     axes.c2p(x_mean - scale * v1[0], y_mean - scale * v1[1]),
        #     axes.c2p(x_mean + scale * v1[0], y_mean + scale * v1[1]),
        #     color=RED,
        #     buff=0,
        #     stroke_width=4
        # )
        
        # # Second singular vector (perpendicular to first)
        # v2 = VT[1]
        # sv2_line = Arrow(
        #     axes.c2p(x_mean - scale * v2[0], y_mean - scale * v2[1]),
        #     axes.c2p(x_mean + scale * v2[0], y_mean + scale * v2[1]),
        #     color=ORANGE,
        #     buff=0,
        #     stroke_width=4
        # )
        
        # # Labels for singular vectors
        # sv1_label = Text("1st component", font_size=24, color=RED).next_to(
        #     axes.c2p(x_mean + scale * v1[0], y_mean + scale * v1[1]), UR
        # )
        # sv2_label = Text("2nd component", font_size=24, color=ORANGE).next_to(
        #     axes.c2p(x_mean + scale * v2[0], y_mean + scale * v2[1]), UL
        # )
        
        # # Animation sequence
        # self.play(Create(axes), Write(x_label), Write(y_label))
        # self.wait(0.5)
        
        # # Show data points
        # self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.1))
        # self.wait(0.5)
        
        # # Show original axes
        # self.play(Create(x_axis_line), Create(y_axis_line))
        # self.wait(0.5)
        
        # # Show singular vectors
        # self.play(Create(sv1_line), Write(sv1_label))
        # self.wait(0.5)
        # self.play(Create(sv2_line), Write(sv2_label))
        # self.wait(1)
        
        # # Fade out original axes
        # self.play(FadeOut(x_axis_line), FadeOut(y_axis_line))
        # self.wait(0.5)
        
        # # Rotate everything so that the first singular vector aligns with x-axis
        # # Calculate rotation angle
        # angle = -np.arctan2(v1[1], v1[0])
        
        # # Create a rotation group
        # rotation_group = VGroup(dots, sv1_line, sv2_line, sv1_label, sv2_label)
        
        # # Rotate around the center point (mean)
        # self.play(
        #     Rotate(rotation_group, angle, about_point=origin),
        #     run_time=2
        # )
        # self.wait(1)
        
        # # Add text explaining the result
        # explanation = Text(
        #     "Data rotated to align with principal component",
        #     font_size=28
        # ).to_edge(UP)
        # self.play(Write(explanation))
        # self.wait(2)

# Render the animation
config.media_width = "100%"
config.pixel_height = 720
config.pixel_width = 1280
config.frame_rate = 30

scene = SVDRotation()
scene.render()

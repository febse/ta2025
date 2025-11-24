"""
Manim animation showing vectors as movement instructions.
Demonstrates how the same vector [1, 2] applied to different starting points
moves each point by the same amount in the same direction.

To render this animation, run:
    manim -pql vector_movement.py VectorMovement
    
For higher quality:
    manim -pqh vector_movement.py VectorMovement
"""

from manim import *
import numpy as np


class VectorMovement(Scene):
    def construct(self):
        # Set up the coordinate system
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 4, 1],
            x_length=6,
            y_length=7,
            axis_config={"color": BLUE},
            tips=False,
        )
        
        # Add axis labels
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Create grid
        grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 4, 1],
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 1,
                "stroke_opacity": 0.3,
            }
        )
        
        # Title
        title = Text("Vectors as Movement Instructions", font_size=36).to_edge(UP, buff=0.7)
        
        # Add axes and title
        # Manual tick labels (avoid add_numbers argument limit in current Manim version)
        x_tick_values = [-3, -2, -1, 1, 2, 3]  # omit 0 to reduce clutter
        y_tick_values = [-3, -2, -1, 1, 2, 3, 4]

        x_labels = VGroup()
        for xv in x_tick_values:
            lbl = Text(str(xv), font_size=18).next_to(axes.c2p(xv, 0), DOWN, buff=0.05)
            x_labels.add(lbl)

        y_labels = VGroup()
        for yv in y_tick_values:
            lbl = Text(str(yv), font_size=18).next_to(axes.c2p(0, yv), LEFT, buff=0.05)
            y_labels.add(lbl)

        self.play(
            Create(grid),
            Create(axes),
            Write(axes_labels),
            *[FadeIn(m) for m in x_labels],
            *[FadeIn(m) for m in y_labels],
            Write(title)
        )
        self.wait()
        
        # Define the vector [1, 2]
        vector_coords = np.array([1, 2, 0])
        
        # Show the vector notation
        vector_text = MathTex(r"\vec{v} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}").to_edge(UP, buff=0.7)
        self.play(Transform(title, vector_text))
        self.wait()
        
        # Starting point 1: [0, 1]
        start_point1 = np.array([0, 1, 0])
        end_point1 = start_point1 + vector_coords
        
        # Starting point 2: [-2, -1]
        start_point2 = np.array([-2, -1, 0])
        end_point2 = start_point2 + vector_coords
        
        # Create dots for starting points
        dot1_start = Dot(axes.c2p(*start_point1[:2]), color=RED, radius=0.08)
        dot2_start = Dot(axes.c2p(*start_point2[:2]), color=GREEN, radius=0.08)
        
        # Labels for starting points
        label1_start = MathTex(r"\begin{bmatrix} 0 \\ 1 \end{bmatrix}", color=RED, font_size=28).next_to(dot1_start, DOWN+RIGHT, buff=0.15)
        label2_start = MathTex(r"\begin{bmatrix} -2 \\ -1 \end{bmatrix}", color=GREEN, font_size=28).next_to(dot2_start, DOWN+LEFT, buff=0.15)
        
        # Show starting points
        self.play(
            FadeIn(dot1_start),
            FadeIn(dot2_start),
            Write(label1_start),
            Write(label2_start)
        )
        self.wait()
        
        # Create instruction text
        instruction = Text("Apply vector [1, 2] to both points", font_size=24).to_edge(DOWN, buff=0.7)
        self.play(Write(instruction))
        self.wait()
        
        # Create vectors from starting points
        vector1 = Arrow(
            axes.c2p(*start_point1[:2]),
            axes.c2p(*end_point1[:2]),
            buff=0,
            color=RED,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.12
        )
        
        vector2 = Arrow(
            axes.c2p(*start_point2[:2]),
            axes.c2p(*end_point2[:2]),
            buff=0,
            color=GREEN,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.12
        )
        
        # Vector labels
        vector1_label = MathTex(r"\vec{v}", color=RED, font_size=28).next_to(vector1, RIGHT, buff=0.1)
        vector2_label = MathTex(r"\vec{v}", color=GREEN, font_size=28).next_to(vector2, LEFT, buff=0.1)
        
        # Animate vectors
        self.play(
            GrowArrow(vector1),
            GrowArrow(vector2),
            Write(vector1_label),
            Write(vector2_label)
        )
        self.wait()
        
        # Create dots for end points
        dot1_end = Dot(axes.c2p(*end_point1[:2]), color=RED, radius=0.08)
        dot2_end = Dot(axes.c2p(*end_point2[:2]), color=GREEN, radius=0.08)
        
        # Labels for end points
        label1_end = MathTex(r"\begin{bmatrix} 1 \\ 3 \end{bmatrix}", color=RED, font_size=28).next_to(dot1_end, UP+RIGHT, buff=0.15)
        label2_end = MathTex(r"\begin{bmatrix} -1 \\ 1 \end{bmatrix}", color=GREEN, font_size=28).next_to(dot2_end, UP+LEFT, buff=0.15)
        
        # Animate movement to end points
        self.play(
            FadeOut(instruction),
            FadeIn(dot1_end),
            FadeIn(dot2_end),
            Write(label1_end),
            Write(label2_end)
        )
        self.wait()
        
        # Show the calculation
        calc_text = Text("Same vector, same displacement!", font_size=28, color=YELLOW).to_edge(DOWN, buff=0.7)
        self.play(Write(calc_text))
        self.wait()
        
        # (Removed stroke highlight enlargement to keep consistent arrow sizes)
        self.wait()
        
        # Show formulas
        formula1 = MathTex(
            r"\begin{bmatrix} 0 \\ 1 \end{bmatrix} + \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 3 \end{bmatrix}",
            color=RED,
            font_size=24
        ).to_corner(UR, buff=0.5)
        
        formula2 = MathTex(
            r"\begin{bmatrix} -2 \\ -1 \end{bmatrix} + \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} -1 \\ 1 \end{bmatrix}",
            color=GREEN,
            font_size=24
        ).next_to(formula1, DOWN, aligned_edge=LEFT, buff=0.3)
        
        self.play(
            FadeOut(calc_text),
            Write(formula1),
            Write(formula2)
        )
        self.wait(2)
        
        # Final emphasis
        conclusion = Text(
            "Vectors translate positions by a fixed amount",
            font_size=28,
            color=YELLOW
        ).to_edge(DOWN, buff=0.7)
        
        self.play(Write(conclusion))
        self.wait(2)
        
        # Now show position vectors from origin and vector sums
        self.play(
            FadeOut(conclusion),
            FadeOut(vector1),
            FadeOut(vector2),
            FadeOut(vector1_label),
            FadeOut(vector2_label),
        )
        
        # Origin point
        origin = np.array([0, 0, 0])
        dot_origin = Dot(axes.c2p(*origin[:2]), color=YELLOW, radius=0.1)
        label_origin = MathTex(r"O", color=YELLOW, font_size=32).next_to(dot_origin, DOWN+LEFT, buff=0.15)
        
        # Position vectors from origin to starting points
        pos_vector1_start = Arrow(
            axes.c2p(*origin[:2]),
            axes.c2p(*start_point1[:2]),
            buff=0,
            color=RED,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.12,
            stroke_opacity=0.6
        )
        
        pos_vector2_start = Arrow(
            axes.c2p(*origin[:2]),
            axes.c2p(*start_point2[:2]),
            buff=0,
            color=GREEN,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.12,
            stroke_opacity=0.6
        )
        
        # Position vectors from origin to end points (vector sum)
        pos_vector1_end = Arrow(
            axes.c2p(*origin[:2]),
            axes.c2p(*end_point1[:2]),
            buff=0,
            color=RED,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.12
        )
        
        pos_vector2_end = Arrow(
            axes.c2p(*origin[:2]),
            axes.c2p(*end_point2[:2]),
            buff=0,
            color=GREEN,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.12
        )
        
        # Show origin and position vectors to starting points
        explanation = Text("Position vectors from origin", font_size=24).to_edge(DOWN, buff=0.7)
        self.play(
            FadeIn(dot_origin),
            Write(label_origin),
            Write(explanation)
        )
        self.wait()
        
        self.play(
            GrowArrow(pos_vector1_start),
            GrowArrow(pos_vector2_start),
        )
        self.wait(2)
        
        # Show vector addition leads to position vectors to end points
        self.play(
            FadeOut(explanation)
        )
        
        sum_explanation = Text(
            "Direct path from origin: Vector sum!",
            font_size=24,
            color=YELLOW
        ).to_edge(DOWN, buff=0.7)
        
        self.play(Write(sum_explanation))
        self.wait()
        
        self.play(
            Transform(pos_vector1_start, pos_vector1_end),
            Transform(pos_vector2_start, pos_vector2_end),
        )
        self.wait(2)
        
        # Show the vector sum formulas
        sum_formula1 = MathTex(
            r"\begin{bmatrix} 0 \\ 1 \end{bmatrix} + \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 3 \end{bmatrix}",
            color=RED,
            font_size=20
        ).next_to(formula2, DOWN, aligned_edge=LEFT, buff=0.5)
        
        sum_formula2 = MathTex(
            r"\begin{bmatrix} -2 \\ -1 \end{bmatrix} + \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} -1 \\ 1 \end{bmatrix}",
            color=GREEN,
            font_size=20
        ).next_to(sum_formula1, DOWN, aligned_edge=LEFT, buff=0.3)
        
        self.play(
            Write(sum_formula1),
            Write(sum_formula2)
        )
        self.wait(3)
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait()


class CosineSimilarity(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 3, 1],
            x_length=10,
            y_length=6,
            axis_config={"color": BLUE, "include_numbers": False, "include_tip": True},
        )
        
        # Add manual tick labels for x-axis (excluding 0)
        x_ticks = [-3, -2, -1, 1, 2, 3]
        for x in x_ticks:
            label = Text(str(x), font_size=20).next_to(axes.c2p(x, 0), DOWN, buff=0.2)
            self.add(label)
        
        # Add manual tick labels for y-axis (excluding 0)
        y_ticks = [-1, 1, 2, 3]
        for y in y_ticks:
            label = Text(str(y), font_size=20).next_to(axes.c2p(0, y), LEFT, buff=0.2)
            self.add(label)
        
        self.play(Create(axes))
        
        # Define vectors
        v = np.array([1, 2, 0])
        w_initial = np.array([-1, 2, 0])
        
        # Create vector arrows
        v_arrow = Arrow(
            axes.c2p(0, 0), 
            axes.c2p(v[0], v[1]), 
            buff=0, 
            color=GREEN,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15
        )
        v_label = MathTex("\\vec{v} = (1, 2)", color=GREEN, font_size=36).next_to(v_arrow.get_end(), RIGHT)
        
        w_arrow = Arrow(
            axes.c2p(0, 0), 
            axes.c2p(w_initial[0], w_initial[1]), 
            buff=0, 
            color=RED,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15
        )
        w_label = MathTex("\\vec{w}", color=RED, font_size=36).next_to(w_arrow.get_end(), LEFT)
        
        self.play(Create(v_arrow), Write(v_label))
        self.play(Create(w_arrow), Write(w_label))
        self.wait()
        
        # Function to compute angle between vectors
        def get_angle(v1, v2):
            dot_product = np.dot(v1[:2], v2[:2])
            mag_v1 = np.linalg.norm(v1[:2])
            mag_v2 = np.linalg.norm(v2[:2])
            cos_theta = dot_product / (mag_v1 * mag_v2)
            cos_theta = np.clip(cos_theta, -1, 1)  # Clamp to valid range
            return np.arccos(cos_theta)
        
        # Function to compute cosine similarity
        def get_cosine_similarity(v1, v2):
            dot_product = np.dot(v1[:2], v2[:2])
            mag_v1 = np.linalg.norm(v1[:2])
            mag_v2 = np.linalg.norm(v2[:2])
            return dot_product / (mag_v1 * mag_v2)
        
        # Create angle arc
        angle = get_angle(v, w_initial)
        arc = Arc(
            radius=0.5,
            start_angle=v_arrow.get_angle(),
            angle=angle if w_initial[0] < 0 else -angle,
            color=YELLOW,
            stroke_width=3
        ).move_arc_center_to(axes.c2p(0, 0))
        
        angle_label = MathTex("\\theta", color=YELLOW, font_size=32).next_to(arc, RIGHT, buff=0.1)
        
        self.play(Create(arc), Write(angle_label))
        self.wait()
        
        # Display angle value
        angle_degrees = angle * 180 / PI
        angle_text = MathTex(
            f"\\theta \\approx {angle_degrees:.1f}^\\circ",
            font_size=32
        ).to_corner(UL)
        
        self.play(Write(angle_text))
        self.wait()
        
        # Display cosine value
        cos_val = get_cosine_similarity(v, w_initial)
        cos_text = MathTex(
            f"\\cos(\\theta) \\approx {cos_val:.3f}",
            font_size=32
        ).next_to(angle_text, DOWN, aligned_edge=LEFT)
        
        self.play(Write(cos_text))
        self.wait()
        
        # Show dot product formula
        dot_product = np.dot(v[:2], w_initial[:2])
        mag_v = np.linalg.norm(v[:2])
        mag_w = np.linalg.norm(w_initial[:2])
        
        formula = MathTex(
            "\\cos(\\theta) = \\frac{\\vec{v} \\cdot \\vec{w}}{||\\vec{v}|| \\, ||\\vec{w}||}",
            font_size=28
        ).next_to(cos_text, DOWN, aligned_edge=LEFT, buff=0.3)
        
        self.play(Write(formula))
        self.wait()
        
        # Show computation
        computation = MathTex(
            f"= \\frac{{({v[0]})({w_initial[0]}) + ({v[1]})({w_initial[1]})}}"
            f"{{\\sqrt{{{v[0]}^2 + {v[1]}^2}} \\cdot \\sqrt{{{w_initial[0]}^2 + {w_initial[1]}^2}}}}",
            font_size=24
        ).next_to(formula, DOWN, aligned_edge=LEFT)
        
        result = MathTex(
            f"= \\frac{{{dot_product:.0f}}}{{{mag_v:.3f} \\cdot {mag_w:.3f}}} \\approx {cos_val:.3f}",
            font_size=24
        ).next_to(computation, DOWN, aligned_edge=LEFT)
        
        self.play(Write(computation))
        self.wait()
        self.play(Write(result))
        self.wait(2)


class MatrixTransform(Scene):
    def construct(self):
        # Axes sized only for up to M^2 v (keep small to leave room for text)
        axes = Axes(
            x_range=[-2, 14, 2],
            y_range=[-2, 14, 2],
            x_length=6,
            y_length=6,
            axis_config={"color": BLUE, "include_numbers": False, "include_tip": True},
        ).to_edge(LEFT, buff=0.8)

        # Manual tick labels (sparser for readability)
        for xv in range(0, 15, 4):
            lbl = Text(str(xv), font_size=18).next_to(axes.c2p(xv, 0), DOWN, buff=0.15)
            if xv != 0:
                self.add(lbl)
        for yv in range(0, 15, 4):
            lbl = Text(str(yv), font_size=18).next_to(axes.c2p(0, yv), LEFT, buff=0.15)
            if yv != 0:
                self.add(lbl)

        title = Text("Matrix Transformation of a Vector", font_size=32).to_edge(UP, buff=0.4)
        self.play(Create(axes), Write(title))
        self.wait()

        # Original vector v and transformation matrix M
        v = np.array([1.0, 2.0])
        M = np.array([[1.95583, 1.0], [1.2, 0.7]])

        # Helper functions
        def norm(x):
            return np.linalg.norm(x)

        def angle_between(a, b):
            cos_val = np.dot(a, b) / (norm(a) * norm(b))
            cos_val = np.clip(cos_val, -1.0, 1.0)
            return np.arccos(cos_val)

        # Draw original vector
        v_arrow = Arrow(
            axes.c2p(0, 0), axes.c2p(v[0], v[1]), buff=0, color=GREEN, stroke_width=4,
            max_tip_length_to_length_ratio=0.12
        )
        v_label = MathTex(r"\vec{v} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}", color=GREEN, font_size=30).next_to(v_arrow.get_end(), RIGHT, buff=0.2)
        self.play(GrowArrow(v_arrow), Write(v_label))
        self.wait()

        # First transformed vector M v
        v1 = M @ v
        t_arrow = Arrow(
            axes.c2p(0, 0), axes.c2p(v1[0], v1[1]), buff=0, color=RED, stroke_width=4,
            max_tip_length_to_length_ratio=0.12
        )
        t_label = MathTex(r"M\vec{v}", color=RED, font_size=30).next_to(t_arrow.get_end(), LEFT, buff=0.2)
        self.play(GrowArrow(t_arrow), Write(t_label))
        self.wait()

        # Angle arc between v and Mv
        theta = angle_between(v, v1)
        start_angle = v_arrow.get_angle()
        # Determine orientation based on sign of cross product (z-component)
        cross_z = v[0]*v1[1] - v[1]*v1[0]
        arc_angle = theta if cross_z > 0 else -theta
        angle_arc = Arc(
            radius=0.8, start_angle=start_angle, angle=arc_angle, color=YELLOW, stroke_width=3
        ).move_arc_center_to(axes.c2p(0, 0))
        angle_tex = MathTex(r"\theta", color=YELLOW, font_size=32).next_to(angle_arc, RIGHT, buff=0.1)
        self.play(Create(angle_arc), Write(angle_tex))
        self.wait()

        # Display magnitudes and cosine
        mag_v = norm(v)
        mag_t = norm(v1)
        cos_val = np.dot(v, v1) / (mag_v * mag_t)
        # Place text to the right of axes
        mag_text = MathTex(
            fr"||\vec{{v}}|| = {mag_v:.3f},\; ||M\vec{{v}}|| = {mag_t:.3f}", font_size=26
        ).next_to(axes, RIGHT, buff=0.8).align_to(axes, UP)
        cos_text = MathTex(
            fr"\cos(\theta) = \frac{{\vec{{v}}\cdot M\vec{{v}}}}{{||\vec{{v}}||\,||M\vec{{v}}||}} = {cos_val:.3f}", font_size=24
        ).next_to(mag_text, DOWN, aligned_edge=LEFT, buff=0.3)
        self.play(Write(mag_text), Write(cos_text))
        self.wait()

        # Dot product computation line
        dot_line = MathTex(
            fr"\vec{{v}}\cdot M\vec{{v}} = ({v[0]:.1f})({v1[0]:.3f}) + ({v[1]:.1f})({v1[1]:.3f}) = {np.dot(v, v1):.3f}",
            font_size=22
        ).next_to(cos_text, DOWN, aligned_edge=LEFT, buff=0.25)
        self.play(Write(dot_line))
        self.wait(2)

        # Prepare iterative multiplication (discrete steps)
        steps = 1  # only one additional multiplication (total two: M v and M^2 v)
        current_vec = v1.copy()

        # Updatable elements (magnitude, cosine, dot, transformed arrow, labels, arc)
        def update_all(new_vec):
            nonlocal angle_arc, angle_tex
            theta_new = angle_between(v, new_vec)
            cross_z_new = v[0]*new_vec[1] - v[1]*new_vec[0]
            arc_angle_new = theta_new if cross_z_new > 0 else -theta_new
            start_angle_new = v_arrow.get_angle()
            new_arc = Arc(
                radius=0.8, start_angle=start_angle_new, angle=arc_angle_new, color=YELLOW, stroke_width=3
            ).move_arc_center_to(axes.c2p(0, 0))
            angle_arc.become(new_arc)
            angle_tex.next_to(angle_arc, RIGHT, buff=0.1)

            mag_new = norm(new_vec)
            cos_new = np.dot(v, new_vec) / (mag_v * mag_new)
            mag_text_new = MathTex(
                fr"||\vec{{v}}|| = {mag_v:.3f},\; ||M^2\vec{{v}}|| = {mag_new:.3f}", font_size=26
            ).next_to(axes, RIGHT, buff=0.8).align_to(axes, UP)
            cos_text_new = MathTex(
                fr"\cos(\theta) = {cos_new:.3f}", font_size=24
            ).next_to(mag_text_new, DOWN, aligned_edge=LEFT, buff=0.3)
            dot_line_new = MathTex(
                fr"\vec{{v}}\cdot M^2\vec{{v}} = {np.dot(v, new_vec):.3f}", font_size=22
            ).next_to(cos_text_new, DOWN, aligned_edge=LEFT, buff=0.25)

            mag_text.become(mag_text_new)
            cos_text.become(cos_text_new)
            dot_line.become(dot_line_new)

        # Iterate discrete multiplications
        for k in range(2, steps + 2):  # only k=2 executed
            next_vec = M @ current_vec
            new_arrow = Arrow(
                axes.c2p(0, 0), axes.c2p(next_vec[0], next_vec[1]), buff=0, color=RED, stroke_width=4,
                max_tip_length_to_length_ratio=0.12
            )
            self.play(Transform(t_arrow, new_arrow), run_time=0.9)
            current_vec = next_vec
            # Update label position
            t_label.next_to(t_arrow.get_end(), LEFT if t_arrow.get_end()[0] > 0 else RIGHT, buff=0.2)
            update_all(current_vec)
            power_tex = MathTex(fr"k = {k}", font_size=22, color=YELLOW).next_to(dot_line, DOWN, aligned_edge=LEFT)
            self.play(Write(power_tex))
            self.wait(0.3)
            self.play(FadeOut(power_tex))

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait()

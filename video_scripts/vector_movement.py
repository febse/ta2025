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


class VectorDefinition(Scene):
    def construct(self):
        # Set up the coordinate system

        x_axis_start = -3
        x_axis_end = 4

        y_axis_start = -3
        y_axis_end = 4

        axes = Axes(
            x_range=[x_axis_start, x_axis_end, 1],
            y_range=[y_axis_start, y_axis_end, 1],
            x_length=6,
            y_length=7,
            axis_config={"color": BLUE},
            tips=False,
        )

        # Create grid with matching size and range
        grid = NumberPlane(
            x_range=[x_axis_start, x_axis_end, 1],
            y_range=[y_axis_start, y_axis_end, 1],
            x_length=6,
            y_length=7,
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 1,
                "stroke_opacity": 0.3,
            }
        )

        # Group grid and axes for perfect alignment
        axes_group = VGroup(grid, axes).next_to(RIGHT, buff=0.5)

        # Add axis labels
        axes_labels = axes.get_axis_labels(
            x_label=Text("BGN", font_size=20),
            y_label=Text("EUR", font_size=20)
        )

        # Title
        title = Text("Bank accounts in a plane", font_size=22).to_corner(UL)

        # Manual tick labels (avoid add_numbers argument limit in current Manim version)
        x_tick_values = [xv for xv in range(x_axis_start, x_axis_end) if xv != 0]
        y_tick_values = [yv for yv in range(y_axis_start, y_axis_end) if yv != 0]

        x_labels = VGroup()

        for xv in x_tick_values:
            lbl = Text(str(xv), font_size=18).next_to(axes.c2p(xv, -0.06), DOWN, buff=0.05)
            x_labels.add(lbl)

        y_labels = VGroup()
        for yv in y_tick_values:
            lbl = Text(str(yv), font_size=18).next_to(axes.c2p(-0.06, yv), LEFT, buff=0.05)
            y_labels.add(lbl)

        self.play(
            Create(axes_group),
            Write(axes_labels),
            *[FadeIn(m) for m in x_labels],
            *[FadeIn(m) for m in y_labels],
            Write(title)
        )
        self.wait()

        # Define the vector [1, 2]
        v = np.array([1, 2, 0])
               
        # # Show the vector notation
        
        # self.play(Transform(title, vector_text))
        self.wait()

        self.play(FadeOut(title))

        instr_deposit_BGN = Text("Deposit 1 BGN, 0 EUR", font_size=24).to_corner(UL, buff=0.7)

        self.play(Write(instr_deposit_BGN))

        # Show a moving arrow from the origin to (1, 0)

        start_point2 = np.array([0, 0, 0])
        end_point2 = start_point2 + np.array([1, 0, 0])

        vector2 = Arrow(
            axes.c2p(*start_point2[:2]),
            axes.c2p(*end_point2[:2]),
            buff=0,
            color=GREEN,
            stroke_width=3,
            tip_length=0.25
        )

        self.play(GrowArrow(vector2))
        self.wait()

        instr_deposit_EUR = Text("Deposit 0 BGN, 2 EUR", font_size=24).next_to(instr_deposit_BGN, DOWN, buff=0.4)
        self.play(Write(instr_deposit_EUR))
        self.wait()

        # Show a moving arrow from (1, 0) to (1, 2)
        start_point3 = end_point2
        end_point3 = start_point3 + np.array([0, 2, 0])
        
        vector3 = Arrow(
            axes.c2p(*start_point3[:2]),
            axes.c2p(*end_point3[:2]),
            buff=0,
            color=BLUE,
            stroke_width=3,
            tip_length=0.25
        )

        self.play(GrowArrow(vector3))
        self.wait()

        vector_v = Arrow(
            axes.c2p(0, 0),
            axes.c2p(*v[:2]),
            buff=0,
            color=YELLOW,
            stroke_width=3,
            tip_length=0.25
        )

        self.play(GrowArrow(vector_v))

        vec_text = rf"\begin{{bmatrix}} {v[0]} \\ {v[1]} \end{{bmatrix}}"

        # Place the vector text next to the endpoint of v
        vector_text_noname = MathTex(vec_text).next_to(
            axes.c2p(*v[:2]), RIGHT + UP, buff=0.2
        )
        vector_text = MathTex(r"\vec{v} = " + vec_text).next_to(instr_deposit_EUR, DOWN, buff=0.5)

        self.play(FadeIn(vector_text_noname), FadeIn(vector_text))

        # # Starting point 1: [0, 0]
        # start_point1 = np.array([0, 0, 0])
        # end_point1 = start_point1 + vector_coords
        
        # # Create dots for starting points
        # dot1_start = Dot(axes.c2p(*start_point1[:2]), color=RED, radius=0.08)
        
        # # Labels for starting points
        # label1_start = MathTex(r"\begin{bmatrix} 0 \\ 0 \end{bmatrix}", color=RED, font_size=28).next_to(dot1_start, DOWN+RIGHT, buff=0.15)
        
        # # Show starting points
        # self.play(
        #     FadeIn(dot1_start),
        #     Write(label1_start)
        # )
        # self.wait()
        
        # # # Create instruction text
        # # instruction = Text("Apply vector [1, 2] to both points", font_size=24).to_edge(DOWN, buff=0.7)
        # # self.play(Write(instruction))
        # # self.wait()
        
        # # # Create vectors from starting points
        # # vector1 = Arrow(
        # #     axes.c2p(*start_point1[:2]),
        # #     axes.c2p(*end_point1[:2]),
        # #     buff=0,
        # #     color=RED,
        # #     stroke_width=3,
        # #     max_tip_length_to_length_ratio=0.12
        # # )
        
        # # # Vector labels
        # # vector1_label = MathTex(r"\vec{v}", color=RED, font_size=28).next_to(vector1, RIGHT, buff=0.1)
        # # vector2_label = MathTex(r"\vec{v}", color=GREEN, font_size=28).next_to(vector2, LEFT, buff=0.1)
        
        # # # Animate vectors
        # # self.play(
        # #     GrowArrow(vector1),
        # #     Write(vector1_label),
        # # )
        # # self.wait()
        
        # # # Create dots for end points
        # # dot1_end = Dot(axes.c2p(*end_point1[:2]), color=RED, radius=0.08)
        # # # Labels for end points
        # # label1_end = MathTex(r"\begin{bmatrix} 1 \\ 3 \end{bmatrix}", color=RED, font_size=28).next_to(dot1_end, UP+RIGHT, buff=0.15)
        
        # # # Animate movement to end points
        # # self.play(
        # #     FadeOut(instruction),
        # #     FadeIn(dot1_end),
        # #     Write(label1_end)
        # # )
        # # self.wait()
        
        # # # Show the calculation
        # # calc_text = Text("Same vector, same displacement!", font_size=28, color=YELLOW).to_edge(DOWN, buff=0.7)
        # # self.play(Write(calc_text))
        # # self.wait()
        
        # # # (Removed stroke highlight enlargement to keep consistent arrow sizes)
        # # self.wait()
        
        # # # Show formulas
        # # formula1 = MathTex(
        # #     r"\begin{bmatrix} 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}",
        # #     color=RED,
        # #     font_size=24
        # # ).to_corner(UR, buff=0.5)
        
        # # formula2 = MathTex(
        # #     r"\begin{bmatrix} -2 \\ -1 \end{bmatrix} + \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} -1 \\ 1 \end{bmatrix}",
        # #     color=GREEN,
        # #     font_size=24
        # # ).next_to(formula1, DOWN, aligned_edge=LEFT, buff=0.3)
        
        # # self.play(
        # #     FadeOut(calc_text),
        # #     Write(formula1),
        # #     Write(formula2)
        # # )
        # # self.wait(2)

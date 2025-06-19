import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

def solve_apollonius_numeric(c1, r1, c2, r2, c3, r3, s1=1, s2=1, s3=1, initial_guess=None):
    """
    Solve the Problem of Apollonius numerically.
    Find circle with center (x,y) and radius r tangent to three given circles.
    s1, s2, s3 = +1 for external tangency, -1 for internal tangency
    """
    
    def equations(vars):
        x, y, r = vars
        # Distance from (x,y) to each circle center should equal sum/difference of radii
        eq1 = np.sqrt((x - c1[0])**2 + (y - c1[1])**2) - (r + s1 * r1)
        eq2 = np.sqrt((x - c2[0])**2 + (y - c2[1])**2) - (r + s2 * r2)
        eq3 = np.sqrt((x - c3[0])**2 + (y - c3[1])**2) - (r + s3 * r3)
        return [eq1, eq2, eq3]
    
    # Initial guess - centroid of the three circles
    if initial_guess is None:
        x0 = (c1[0] + c2[0] + c3[0]) / 3
        y0 = (c1[1] + c2[1] + c3[1]) / 3
        r0 = (r1 + r2 + r3) / 3
        initial_guess = [x0, y0, r0]
    
    try:
        solution = fsolve(equations, initial_guess)
        x, y, r = solution
        
        # Verify the solution
        residual = equations(solution)
        if np.max(np.abs(residual)) < 1e-6 and r > 0:
            return np.array([x, y]), r
        else:
            return None, None
    except:
        return None, None

def find_all_apollonius_circles(c1, r1, c2, r2, c3, r3):
    """
    Find all possible Apollonius circles (up to 8 solutions).
    Each solution corresponds to a different combination of internal/external tangencies.
    """
    solutions = []
    
    # Try all 8 combinations of internal (-1) and external (+1) tangencies
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            for s3 in [1, -1]:
                # Skip cases where we try to be internal to a circle smaller than the solution
                center, radius = solve_apollonius_numeric(c1, r1, c2, r2, c3, r3, s1, s2, s3)
                
                if center is not None and radius is not None:
                    # Check if this is internal tangency to a smaller circle (invalid)
                    valid = True
                    if s1 == -1 and radius > r1:
                        valid = False
                    if s2 == -1 and radius > r2:
                        valid = False
                    if s3 == -1 and radius > r3:
                        valid = False
                    
                    if valid:
                        solutions.append({
                            'center': center,
                            'radius': radius,
                            'tangency': (s1, s2, s3),
                            'type': f"{'E' if s1==1 else 'I'}{'E' if s2==1 else 'I'}{'E' if s3==1 else 'I'}"
                        })
    
    return solutions

def plot_apollonius(c1, r1, c2, r2, c3, r3, solutions, selected_solution=None):
    """Plot the three given circles and the Apollonius circle solutions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Plot the three given circles
    circle1 = Circle(c1, r1, fill=False, edgecolor='blue', linewidth=2, label='Circle 1')
    circle2 = Circle(c2, r2, fill=False, edgecolor='green', linewidth=2, label='Circle 2')
    circle3 = Circle(c3, r3, fill=False, edgecolor='red', linewidth=2, label='Circle 3')
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    
    # Plot centers
    ax.plot(*c1, 'bo', markersize=8)
    ax.plot(*c2, 'go', markersize=8)
    ax.plot(*c3, 'ro', markersize=8)
    
    # Add labels
    ax.text(c1[0], c1[1], '  1', fontsize=12, ha='left', va='center')
    ax.text(c2[0], c2[1], '  2', fontsize=12, ha='left', va='center')
    ax.text(c3[0], c3[1], '  3', fontsize=12, ha='left', va='center')
    
    # Plot all solutions
    colors = plt.cm.rainbow(np.linspace(0, 1, len(solutions)))
    
    for i, (sol, color) in enumerate(zip(solutions, colors)):
        if selected_solution is None or i == selected_solution:
            alpha = 0.8 if selected_solution is None else 1.0
            linewidth = 2 if selected_solution is None else 3
            
            solution_circle = Circle(sol['center'], sol['radius'], 
                                   fill=False, edgecolor=color, 
                                   linewidth=linewidth, alpha=alpha,
                                   linestyle='-' if 'E' in sol['type'] else '--')
            ax.add_patch(solution_circle)
            
            # Add center point
            ax.plot(*sol['center'], 'o', color=color, markersize=6)
            
            # Add label
            ax.text(sol['center'][0], sol['center'][1], 
                   f"\n  {sol['type']}", fontsize=10, ha='center', va='top')
    
    # Set axis limits
    all_centers = [c1, c2, c3] + [sol['center'] for sol in solutions]
    all_radii = [r1, r2, r3] + [sol['radius'] for sol in solutions]
    
    x_coords = [c[0] for c in all_centers]
    y_coords = [c[1] for c in all_centers]
    
    margin = max(all_radii) * 1.5
    ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
    ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Apollonius Circles: Finding Circles Tangent to Three Given Circles', fontsize=14)
    
    # Add legend explaining tangency types
    ax.text(0.02, 0.98, 'E = External tangency\nI = Internal tangency\nSolid = All external\nDashed = Some internal', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

# Streamlit App
st.title("🔵 Apollonius Circle Solver")
st.markdown("### Find all circles tangent to three given circles")

st.markdown("""
The **Problem of Apollonius** asks: Given three circles, find all circles that are tangent to all three.

There can be up to **8 different solutions**, depending on whether the tangency is:
- **External (E)**: The solution circle touches the given circle from outside
- **Internal (I)**: The solution circle contains the given circle and touches from inside
""")

# Sidebar for circle inputs
st.sidebar.header("Define Three Circles")

col1, col2 = st.sidebar.columns(2)

# Circle 1
st.sidebar.markdown("**Circle 1** 🔵")
with col1:
    x1 = st.number_input("Center X₁", value=0.0, step=0.5, key="x1")
    r1 = st.number_input("Radius r₁", value=3.0, min_value=0.1, step=0.5, key="r1")
with col2:
    y1 = st.number_input("Center Y₁", value=0.0, step=0.5, key="y1")

# Circle 2
st.sidebar.markdown("**Circle 2** 🟢")
with col1:
    x2 = st.number_input("Center X₂", value=5.0, step=0.5, key="x2")
    r2 = st.number_input("Radius r₂", value=2.0, min_value=0.1, step=0.5, key="r2")
with col2:
    y2 = st.number_input("Center Y₂", value=0.0, step=0.5, key="y2")

# Circle 3
st.sidebar.markdown("**Circle 3** 🔴")
with col1:
    x3 = st.number_input("Center X₃", value=2.5, step=0.5, key="x3")
    r3 = st.number_input("Radius r₃", value=2.5, min_value=0.1, step=0.5, key="r3")
with col2:
    y3 = st.number_input("Center Y₃", value=4.0, step=0.5, key="y3")

st.sidebar.markdown("---")

# Preset examples
if st.sidebar.button("Load Example 1: Classic"):
    st.session_state.x1, st.session_state.y1, st.session_state.r1 = -3.0, 0.0, 2.0
    st.session_state.x2, st.session_state.y2, st.session_state.r2 = 3.0, 0.0, 2.0
    st.session_state.x3, st.session_state.y3, st.session_state.r3 = 0.0, 3.0, 1.5
    st.rerun()

if st.sidebar.button("Load Example 2: Nested"):
    st.session_state.x1, st.session_state.y1, st.session_state.r1 = 0.0, 0.0, 5.0
    st.session_state.x2, st.session_state.y2, st.session_state.r2 = 2.0, 0.0, 1.0
    st.session_state.x3, st.session_state.y3, st.session_state.r3 = -1.0, 1.0, 0.8
    st.rerun()

# Solve button
if st.sidebar.button("🔍 Find Tangent Circles", type="primary"):
    # Define circles
    c1 = np.array([x1, y1])
    c2 = np.array([x2, y2])
    c3 = np.array([x3, y3])
    
    # Find all solutions
    with st.spinner("Solving Apollonius problem..."):
        solutions = find_all_apollonius_circles(c1, r1, c2, r2, c3, r3)
    
    if solutions:
        st.success(f"Found {len(solutions)} solution(s)!")
        
        # Create plot
        fig = plot_apollonius(c1, r1, c2, r2, c3, r3, solutions)
        st.pyplot(fig)
        
        # Display solution details
        st.subheader("Solution Details")
        
        for i, sol in enumerate(solutions):
            with st.expander(f"Solution {i+1}: {sol['type']} (click to expand)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Center X", f"{sol['center'][0]:.3f}")
                    st.metric("Center Y", f"{sol['center'][1]:.3f}")
                
                with col2:
                    st.metric("Radius", f"{sol['radius']:.3f}")
                    
                with col3:
                    tangency_types = []
                    if sol['tangency'][0] == 1:
                        tangency_types.append("External to 1")
                    else:
                        tangency_types.append("Internal to 1")
                    if sol['tangency'][1] == 1:
                        tangency_types.append("External to 2")
                    else:
                        tangency_types.append("Internal to 2")
                    if sol['tangency'][2] == 1:
                        tangency_types.append("External to 3")
                    else:
                        tangency_types.append("Internal to 3")
                    
                    st.write("**Tangency Types:**")
                    for t in tangency_types:
                        st.write(f"• {t}")
                
                # Verification
                st.write("**Distance Verification:**")
                d1 = np.linalg.norm(sol['center'] - c1)
                d2 = np.linalg.norm(sol['center'] - c2)
                d3 = np.linalg.norm(sol['center'] - c3)
                
                expected1 = sol['radius'] + sol['tangency'][0] * r1
                expected2 = sol['radius'] + sol['tangency'][1] * r2
                expected3 = sol['radius'] + sol['tangency'][2] * r3
                
                st.write(f"• Distance to circle 1: {d1:.3f} (expected: {expected1:.3f})")
                st.write(f"• Distance to circle 2: {d2:.3f} (expected: {expected2:.3f})")
                st.write(f"• Distance to circle 3: {d3:.3f} (expected: {expected3:.3f})")
    else:
        st.error("No solutions found. The circles might be in a configuration with no tangent circle.")

# Theory section
with st.expander("📚 Mathematical Background"):
    st.markdown("""
    ### The Problem of Apollonius
    
    Given three circles in the plane, find all circles that are tangent to all three. 
    This problem was posed by Apollonius of Perga around 200 BC.
    
    ### Solution Types
    
    Each solution is characterized by three letters (E or I):
    - **EEE**: External to all three circles
    - **EEI**: External to circles 1 and 2, internal to circle 3
    - **EII**: External to circle 1, internal to circles 2 and 3
    - **III**: Internal to all three circles (only possible if one circle contains the other two)
    
    ### Mathematical Approach
    
    For a solution circle with center (x, y) and radius r:
    
    The distance from its center to circle i's center must equal:
    - r + rᵢ (for external tangency)
    - |r - rᵢ| (for internal tangency)
    
    This gives us three equations:
    - √[(x-x₁)² + (y-y₁)²] = r ± r₁
    - √[(x-x₂)² + (y-y₂)²] = r ± r₂
    - √[(x-x₃)² + (y-y₃)²] = r ± r₃
    
    We solve this system numerically for each combination of ± signs.
    
    ### Special Cases
    
    - If the three circles are mutually tangent, one solution is the circle through their points of tangency
    - If one circle contains the other two, some internal tangency solutions may not exist
    - Degenerate cases can produce lines (circles with infinite radius) as solutions
    """)

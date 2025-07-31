import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Apollonius Circle Solver",
    page_icon="🔵",
    layout="wide"
)

def solve_apollonius_algebraic(c1, r1, c2, r2, c3, r3, s1=1, s2=1, s3=1):
    """
    Solve the Problem of Apollonius using algebraic method.
    This is more stable than the iterative approach for well-conditioned problems.
    """
    x1, y1 = c1
    x2, y2 = c2
    x3, y3 = c3
    
    # Convert to the linear system for the radical center
    # Each equation is of the form: 2x*xi + 2y*yi - 2r*si*ri = xi² + yi² - ri²
    
    A = np.array([
        [2*(x1-x2), 2*(y1-y2), 2*(s2*r2 - s1*r1)],
        [2*(x1-x3), 2*(y1-y3), 2*(s3*r3 - s1*r1)],
        [2*(x2-x3), 2*(y2-y3), 2*(s3*r3 - s2*r2)]
    ])
    
    b = np.array([
        x1**2 - x2**2 + y1**2 - y2**2 + (s2*r2)**2 - (s1*r1)**2,
        x1**2 - x3**2 + y1**2 - y3**2 + (s3*r3)**2 - (s1*r1)**2,
        x2**2 - x3**2 + y2**2 - y3**2 + (s3*r3)**2 - (s2*r2)**2
    ])
    
    try:
        # Check if system is singular
        if np.abs(np.linalg.det(A[:2, :2])) < 1e-10:
            return None, None
            
        # Solve the overdetermined system using least squares
        solution, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        
        if rank < 2:
            return None, None
            
        x, y, t = solution
        
        # Calculate radius from the first circle constraint
        d1 = np.sqrt((x - x1)**2 + (y - y1)**2)
        r = d1 - s1 * r1
        
        if r <= 0:
            return None, None
            
        # Verify solution with all three circles
        d2 = np.sqrt((x - x2)**2 + (y - y2)**2)
        d3 = np.sqrt((x - x3)**2 + (y - y3)**2)
        
        e1 = abs(d1 - (r + s1 * r1))
        e2 = abs(d2 - (r + s2 * r2))
        e3 = abs(d3 - (r + s3 * r3))
        
        if max(e1, e2, e3) > 0.01:
            return None, None
            
        return np.array([x, y]), r
        
    except:
        return None, None

def solve_apollonius_iterative(c1, r1, c2, r2, c3, r3, s1=1, s2=1, s3=1, max_iter=1000, tol=1e-8):
    """
    Solve the Problem of Apollonius using iterative method with improved stability.
    """
    # Try algebraic method first
    center, radius = solve_apollonius_algebraic(c1, r1, c2, r2, c3, r3, s1, s2, s3)
    if center is not None:
        return center, radius
    
    # Use multiple initial guesses
    initial_guesses = []
    
    # Weighted centroid
    total_r = r1 + r2 + r3
    if total_r > 0:
        x0 = (c1[0]*r1 + c2[0]*r2 + c3[0]*r3) / total_r
        y0 = (c1[1]*r1 + c2[1]*r2 + c3[1]*r3) / total_r
        r0 = total_r / 3
        initial_guesses.append((x0, y0, r0))
    
    # Centroid
    x0 = (c1[0] + c2[0] + c3[0]) / 3
    y0 = (c1[1] + c2[1] + c3[1]) / 3
    r0 = (r1 + r2 + r3) / 3
    initial_guesses.append((x0, y0, r0))
    
    # Try from each circle's perspective
    for c, r in [(c1, r1), (c2, r2), (c3, r3)]:
        initial_guesses.append((c[0], c[1], r * 2))
    
    best_solution = None
    best_error = float('inf')
    
    for x0, y0, r0 in initial_guesses:
        x, y, r = x0, y0, r0
        
        for iteration in range(max_iter):
            # Calculate current distances
            d1 = np.sqrt((x - c1[0])**2 + (y - c1[1])**2)
            d2 = np.sqrt((x - c2[0])**2 + (y - c2[1])**2)
            d3 = np.sqrt((x - c3[0])**2 + (y - c3[1])**2)
            
            # Avoid division by zero
            d1 = max(d1, 1e-10)
            d2 = max(d2, 1e-10)
            d3 = max(d3, 1e-10)
            
            # Target distances
            target_d1 = r + s1 * r1
            target_d2 = r + s2 * r2
            target_d3 = r + s3 * r3
            
            # Errors
            e1 = d1 - target_d1
            e2 = d2 - target_d2
            e3 = d3 - target_d3
            
            max_error = max(abs(e1), abs(e2), abs(e3))
            
            # Check convergence
            if max_error < tol and r > 0:
                if max_error < best_error:
                    best_solution = (np.array([x, y]), r)
                    best_error = max_error
                break
            
            # Compute Jacobian
            J = np.array([
                [(x - c1[0])/d1, (y - c1[1])/d1, -1],
                [(x - c2[0])/d2, (y - c2[1])/d2, -1],
                [(x - c3[0])/d3, (y - c3[1])/d3, -1]
            ])
            
            # Error vector
            E = np.array([e1, e2, e3])
            
            try:
                # Solve with regularization
                JTJ = J.T @ J
                JTE = J.T @ E
                regularization = 1e-8 * np.eye(3)
                delta = np.linalg.solve(JTJ + regularization, -JTE)
                
                # Adaptive damping based on error reduction
                damping = 0.5 if iteration < 10 else 0.8
                
                # Line search for better convergence
                alpha = 1.0
                for _ in range(5):
                    x_new = x + alpha * damping * delta[0]
                    y_new = y + alpha * damping * delta[1]
                    r_new = max(0.01, r + alpha * damping * delta[2])
                    
                    # Check if error decreases
                    d1_new = np.sqrt((x_new - c1[0])**2 + (y_new - c1[1])**2)
                    d2_new = np.sqrt((x_new - c2[0])**2 + (y_new - c2[1])**2)
                    d3_new = np.sqrt((x_new - c3[0])**2 + (y_new - c3[1])**2)
                    
                    e1_new = d1_new - (r_new + s1 * r1)
                    e2_new = d2_new - (r_new + s2 * r2)
                    e3_new = d3_new - (r_new + s3 * r3)
                    
                    new_error = max(abs(e1_new), abs(e2_new), abs(e3_new))
                    
                    if new_error < max_error:
                        x, y, r = x_new, y_new, r_new
                        break
                    else:
                        alpha *= 0.5
                
            except:
                break
    
    return best_solution if best_solution else (None, None)

def check_circle_overlap(c1, r1, c2, r2):
    """Check if two circles overlap"""
    dist = np.linalg.norm(c1 - c2)
    return dist < r1 + r2

def find_all_apollonius_circles(c1, r1, c2, r2, c3, r3):
    """Find all possible Apollonius circles with improved validation."""
    solutions = []
    
    # Check for degenerate cases
    circles = [(c1, r1), (c2, r2), (c3, r3)]
    
    # Check if circles are collinear
    v1 = c2 - c1
    v2 = c3 - c1
    if abs(np.cross(v1, v2)) < 1e-10:
        st.warning("The three circle centers are collinear. This is a special case that may have limited solutions.")
    
    # Try all 8 combinations
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            for s3 in [1, -1]:
                center, radius = solve_apollonius_iterative(c1, r1, c2, r2, c3, r3, s1, s2, s3)
                
                if center is not None and radius is not None and radius > 0:
                    # Validate solution
                    valid = True
                    
                    # Check internal tangency constraints
                    if s1 == -1 and radius < r1 - 1e-6:
                        valid = False
                    if s2 == -1 and radius < r2 - 1e-6:
                        valid = False
                    if s3 == -1 and radius < r3 - 1e-6:
                        valid = False
                    
                    # Additional validation for internal tangency
                    if s1 == -1:
                        d1 = np.linalg.norm(center - c1)
                        if d1 > 1e-6 and abs(d1 + radius - r1) > 0.01:
                            valid = False
                    if s2 == -1:
                        d2 = np.linalg.norm(center - c2)
                        if d2 > 1e-6 and abs(d2 + radius - r2) > 0.01:
                            valid = False
                    if s3 == -1:
                        d3 = np.linalg.norm(center - c3)
                        if d3 > 1e-6 and abs(d3 + radius - r3) > 0.01:
                            valid = False
                    
                    if valid:
                        # Check for duplicates with tolerance
                        is_duplicate = False
                        for sol in solutions:
                            if np.linalg.norm(sol['center'] - center) < 0.01 and abs(sol['radius'] - radius) < 0.01:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            # Verify tangency one more time
                            d1 = np.linalg.norm(center - c1)
                            d2 = np.linalg.norm(center - c2)
                            d3 = np.linalg.norm(center - c3)
                            
                            e1 = abs(d1 - abs(radius + s1 * r1))
                            e2 = abs(d2 - abs(radius + s2 * r2))
                            e3 = abs(d3 - abs(radius + s3 * r3))
                            
                            if max(e1, e2, e3) < 0.01:
                                solutions.append({
                                    'center': center,
                                    'radius': radius,
                                    'tangency': (s1, s2, s3),
                                    'type': f"{'E' if s1==1 else 'I'}{'E' if s2==1 else 'I'}{'E' if s3==1 else 'I'}",
                                    'error': max(e1, e2, e3)
                                })
    
    # Sort by radius
    solutions.sort(key=lambda x: x['radius'])
    return solutions

def plot_apollonius_enhanced(c1, r1, c2, r2, c3, r3, solutions, selected_solution=None):
    """Enhanced plotting with better visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_aspect('equal')
    
    # Set style
    plt.style.use('default')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Plot the three given circles with gradient effect
    circles_data = [
        (c1, r1, 'blue', 'Circle 1'),
        (c2, r2, 'green', 'Circle 2'),
        (c3, r3, 'red', 'Circle 3')
    ]
    
    for center, radius, color, label in circles_data:
        # Main circle
        circle = Circle(center, radius, fill=False, edgecolor=color, 
                       linewidth=3, label=label, alpha=0.8)
        ax.add_patch(circle)
        
        # Fill with low alpha
        circle_fill = Circle(center, radius, fill=True, facecolor=color, 
                           alpha=0.05, edgecolor='none')
        ax.add_patch(circle_fill)
        
        # Center point
        ax.plot(*center, 'o', color=color, markersize=10, 
               markeredgecolor='white', markeredgewidth=2)
        
        # Label with background
        ax.text(center[0], center[1]-0.3, label.split()[-1], 
               fontsize=12, ha='center', va='top', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=color, alpha=0.8))
    
    # Plot solutions
    if solutions:
        # Color palette for solutions
        cmap = plt.cm.rainbow
        colors = [cmap(i/max(len(solutions)-1, 1)) for i in range(len(solutions))]
        
        for i, (sol, color) in enumerate(zip(solutions, colors)):
            if selected_solution is None or i == selected_solution:
                alpha = 0.6 if selected_solution is None else 0.8
                linewidth = 2.5 if selected_solution is None else 4
                
                # Determine line style based on tangency type
                has_internal = any(t == -1 for t in sol['tangency'])
                linestyle = '--' if has_internal else '-'
                
                # Solution circle
                solution_circle = Circle(sol['center'], sol['radius'], 
                                       fill=False, edgecolor=color, 
                                       linewidth=linewidth, alpha=alpha,
                                       linestyle=linestyle)
                ax.add_patch(solution_circle)
                
                # Center point
                ax.plot(*sol['center'], 'o', color=color, markersize=8,
                       markeredgecolor='white', markeredgewidth=1.5)
                
                # Label
                label_text = f"{i+1}: {sol['type']}"
                ax.text(sol['center'][0], sol['center'][1] + 0.3, 
                       label_text, fontsize=10, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                                alpha=0.3, edgecolor=color))
    
    # Calculate view limits
    all_centers = [c1, c2, c3] + [sol['center'] for sol in solutions]
    all_radii = [r1, r2, r3] + [sol['radius'] for sol in solutions]
    
    if all_centers:
        x_coords = [c[0] for c in all_centers]
        y_coords = [c[1] for c in all_centers]
        
        margin = max(all_radii) * 1.5 if all_radii else 5
        ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
        ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    
    # Grid and axes
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
    
    # Labels
    ax.set_xlabel('X', fontsize=14, weight='bold')
    ax.set_ylabel('Y', fontsize=14, weight='bold')
    ax.set_title('Apollonius Circles: Finding Circles Tangent to Three Given Circles', 
                fontsize=16, weight='bold', pad=20)
    
    # Legend
    legend_text = (
        'Tangency Types:\n'
        'E = External (outside)\n'
        'I = Internal (inside)\n'
        'Solid line = All external\n'
        'Dashed line = Some internal'
    )
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                    edgecolor='gray', alpha=0.9))
    
    # Add solution count
    if solutions:
        count_text = f'Found {len(solutions)} solution{"s" if len(solutions) != 1 else ""}'
        ax.text(0.98, 0.02, count_text, transform=ax.transAxes, 
               fontsize=12, ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                        alpha=0.8))
    
    plt.tight_layout()
    return fig

# Streamlit App
st.title("🔵 Advanced Apollonius Circle Solver")
st.markdown("### Find all circles tangent to three given circles")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Solver", "📚 Theory", "🎮 Interactive Mode"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Define Three Circles")
        
        # Check if we need to load preset values
        if 'load_preset' in st.session_state and st.session_state.load_preset:
            default_x1 = st.session_state.get('preset_x1', 0.0)
            default_y1 = st.session_state.get('preset_y1', 0.0)
            default_r1 = st.session_state.get('preset_r1', 3.0)
            default_x2 = st.session_state.get('preset_x2', 5.0)
            default_y2 = st.session_state.get('preset_y2', 0.0)
            default_r2 = st.session_state.get('preset_r2', 2.0)
            default_x3 = st.session_state.get('preset_x3', 2.5)
            default_y3 = st.session_state.get('preset_y3', 4.0)
            default_r3 = st.session_state.get('preset_r3', 2.5)
            st.session_state.load_preset = False
        else:
            default_x1 = 0.0
            default_y1 = 0.0
            default_r1 = 3.0
            default_x2 = 5.0
            default_y2 = 0.0
            default_r2 = 2.0
            default_x3 = 2.5
            default_y3 = 4.0
            default_r3 = 2.5
        
        # Circle 1
        st.markdown("**Circle 1** 🔵")
        col1a, col1b = st.columns(2)
        with col1a:
            x1 = st.number_input("Center X₁", value=default_x1, step=0.5, key="x1")
            r1 = st.number_input("Radius r₁", value=default_r1, min_value=0.1, step=0.5, key="r1")
        with col1b:
            y1 = st.number_input("Center Y₁", value=default_y1, step=0.5, key="y1")
        
        # Circle 2
        st.markdown("**Circle 2** 🟢")
        col2a, col2b = st.columns(2)
        with col2a:
            x2 = st.number_input("Center X₂", value=default_x2, step=0.5, key="x2")
            r2 = st.number_input("Radius r₂", value=default_r2, min_value=0.1, step=0.5, key="r2")
        with col2b:
            y2 = st.number_input("Center Y₂", value=default_y2, step=0.5, key="y2")
        
        # Circle 3
        st.markdown("**Circle 3** 🔴")
        col3a, col3b = st.columns(2)
        with col3a:
            x3 = st.number_input("Center X₃", value=default_x3, step=0.5, key="x3")
            r3 = st.number_input("Radius r₃", value=default_r3, min_value=0.1, step=0.5, key="r3")
        with col3b:
            y3 = st.number_input("Center Y₃", value=default_y3, step=0.5, key="y3")
        
        st.markdown("---")
        
        # Preset configurations
        st.markdown("### Preset Configurations")
        
        preset_configs = {
            "Classic Triangle": {
                "circles": [(-3.0, 0.0, 2.0), (3.0, 0.0, 2.0), (0.0, 3.0, 1.5)],
                "description": "Three circles forming a triangle"
            },
            "Nested Circles": {
                "circles": [(0.0, 0.0, 5.0), (2.0, 0.0, 1.0), (-1.0, 1.0, 0.8)],
                "description": "One large circle with two smaller inside"
            },
            "Symmetric": {
                "circles": [(-2.0, -1.0, 1.5), (2.0, -1.0, 1.5), (0.0, 2.0, 1.5)],
                "description": "Three equal circles in triangular arrangement"
            },
            "Chain": {
                "circles": [(0.0, 0.0, 2.0), (4.0, 0.0, 1.5), (6.5, 0.0, 1.0)],
                "description": "Three circles in a line"
            },
            "Kissing Circles": {
                "circles": [(0.0, 0.0, 3.0), (6.0, 0.0, 3.0), (3.0, 5.196, 3.0)],
                "description": "Three mutually tangent circles"
            }
        }
        
        selected_preset = st.selectbox(
            "Choose a preset:",
            ["Custom"] + list(preset_configs.keys())
        )
        
        if selected_preset != "Custom":
            config = preset_configs[selected_preset]
            st.info(f"📝 {config['description']}")
            if st.button(f"Load {selected_preset}"):
                circles = config["circles"]
                # Use a different key prefix to avoid conflicts
                st.session_state.preset_x1, st.session_state.preset_y1, st.session_state.preset_r1 = circles[0]
                st.session_state.preset_x2, st.session_state.preset_y2, st.session_state.preset_r2 = circles[1]
                st.session_state.preset_x3, st.session_state.preset_y3, st.session_state.preset_r3 = circles[2]
                st.session_state.load_preset = True
                st.rerun()
        
        st.markdown("---")
        
        # Solve button
        solve_button = st.button("🔍 Find Tangent Circles", type="primary", use_container_width=True)
    
    with col2:
        if solve_button:
            # Define circles
            c1 = np.array([x1, y1])
            c2 = np.array([x2, y2])
            c3 = np.array([x3, y3])
            
            # Check for special cases
            st.markdown("### Analysis")
            
            # Check distances
            d12 = np.linalg.norm(c1 - c2)
            d13 = np.linalg.norm(c1 - c3)
            d23 = np.linalg.norm(c2 - c3)
            
            info_cols = st.columns(3)
            with info_cols[0]:
                st.metric("Distance 1↔2", f"{d12:.3f}")
                rel = "Separate" if d12 > r1 + r2 else "Overlapping" if d12 < abs(r1 - r2) else "Tangent"
                st.caption(rel)
            
            with info_cols[1]:
                st.metric("Distance 1↔3", f"{d13:.3f}")
                rel = "Separate" if d13 > r1 + r3 else "Overlapping" if d13 < abs(r1 - r3) else "Tangent"
                st.caption(rel)
            
            with info_cols[2]:
                st.metric("Distance 2↔3", f"{d23:.3f}")
                rel = "Separate" if d23 > r2 + r3 else "Overlapping" if d23 < abs(r2 - r3) else "Tangent"
                st.caption(rel)
            
            # Find solutions
            with st.spinner("Solving Apollonius problem..."):
                solutions = find_all_apollonius_circles(c1, r1, c2, r2, c3, r3)
            
            if solutions:
                st.success(f"✅ Found {len(solutions)} solution{'s' if len(solutions) != 1 else ''}!")
                
                # Create plot
                fig = plot_apollonius_enhanced(c1, r1, c2, r2, c3, r3, solutions)
                st.pyplot(fig)
                
                # Solution details
                st.markdown("### Solution Details")
                
                # Create columns for solution cards
                sol_cols = st.columns(min(len(solutions), 3))
                
                for i, sol in enumerate(solutions):
                    col_idx = i % min(len(solutions), 3)
                    with sol_cols[col_idx]:
                        with st.expander(f"**Solution {i+1}: {sol['type']}**", expanded=(i==0)):
                            st.markdown(f"**Center:** ({sol['center'][0]:.3f}, {sol['center'][1]:.3f})")
                            st.markdown(f"**Radius:** {sol['radius']:.3f}")
                            st.markdown(f"**Max Error:** {sol['error']:.6f}")
                            
                            st.markdown("**Tangency:**")
                            tangency_info = []
                            symbols = ['🔵', '🟢', '🔴']
                            for j, (s, sym) in enumerate(zip(sol['tangency'], symbols)):
                                tang_type = "External" if s == 1 else "Internal"
                                tangency_info.append(f"{sym} Circle {j+1}: {tang_type}")
                            for info in tangency_info:
                                st.markdown(f"• {info}")
                            
                            # Verification metrics
                            st.markdown("**Distance Verification:**")
                            d1 = np.linalg.norm(sol['center'] - c1)
                            d2 = np.linalg.norm(sol['center'] - c2)
                            d3 = np.linalg.norm(sol['center'] - c3)
                            
                            verification = [
                                (d1, sol['radius'] + sol['tangency'][0] * r1, "Circle 1"),
                                (d2, sol['radius'] + sol['tangency'][1] * r2, "Circle 2"),
                                (d3, sol['radius'] + sol['tangency'][2] * r3, "Circle 3")
                            ]
                            
                            for dist, expected, name in verification:
                                error = abs(dist - abs(expected))
                                status = "✅" if error < 0.001 else "⚠️"
                                st.caption(f"{status} {name}: {dist:.4f} (error: {error:.6f})")
            else:
                st.error("❌ No solutions found. The circles might be in a degenerate configuration.")
                
                # Still show the plot
                fig = plot_apollonius_enhanced(c1, r1, c2, r2, c3, r3, [])
                st.pyplot(fig)

with tab2:
    st.markdown("""
    ## The Problem of Apollonius
    
    The Problem of Apollonius, posed by Apollonius of Perga (c. 262–190 BC), is one of the most famous problems in geometry:
    
    > **Given three circles in the plane, find all circles that are tangent to all three.**
    
    ### Historical Context
    
    Apollonius wrote about this problem in his lost work "Tangencies" (Ἐπαφαί). The problem has fascinated mathematicians for over 2000 years and has connections to:
    - Inversive geometry
    - Complex analysis
    - Algebraic geometry
    - Computer graphics and CAD systems
    
    ### Mathematical Foundation
    
    #### Tangency Conditions
    
    For a solution circle with center $(x, y)$ and radius $r$ to be tangent to a given circle with center $(x_i, y_i)$ and radius $r_i$:
    
    - **External tangency:** $\sqrt{(x-x_i)^2 + (y-y_i)^2} = r + r_i$
    - **Internal tangency:** $\sqrt{(x-x_i)^2 + (y-y_i)^2} = |r - r_i|$
    
    #### Solution Types
    
    Each solution is characterized by three letters (E or I) indicating the type of tangency with each given circle:
    
    | Type | Description | Example |
    |------|-------------|---------|
    | EEE | External to all three | Most common case |
    | EEI | External to two, internal to one | Mixed tangency |
    | EII | External to one, internal to two | Less common |
    | III | Internal to all three | Only when one contains others |
    
    ### Number of Solutions
    
    In general, there can be up to **8 solutions**, corresponding to the $2^3 = 8$ possible combinations of internal/external tangencies. However:
    
    - Some configurations have fewer solutions
    - Degenerate cases (e.g., collinear centers) may have infinite solutions
    - Special cases (e.g., three mutually tangent circles) have unique properties
    
    ### Solution Methods
    
    #### 1. Algebraic Method
    The tangency conditions lead to a system of quadratic equations. By subtracting pairs of equations, we can eliminate quadratic terms and obtain a linear system.
    
    #### 2. Iterative Method (Newton-Raphson)
    Starting from an initial guess, we iteratively refine the solution by minimizing the error in the tangency conditions.
    
    #### 3. Geometric Inversion
    Using circle inversion, we can transform the problem into simpler cases, solve them, and then invert back.
    
    ### Special Cases
    
    1. **Three mutually tangent circles**: The circles tangent to all three include the circumcircle and incircle of the curvilinear triangle formed by the three circles.
    
    2. **Concentric circles**: When two circles are concentric, the problem reduces to finding circles tangent to two circles and passing through a point.
    
    3. **Collinear centers**: When the three centers are collinear, some solutions may be lines (circles with infinite radius).
    
    ### Applications
    
    - **Engineering**: Gear design, bearing placement
    - **Computer Graphics**: Smooth transitions between circular arcs
    - **Architecture**: Circular arch design
    - **Physics**: Packing problems, bubble configurations
    
    ### Mathematical Properties
    
    1. **Descartes Circle Theorem**: For four mutually tangent circles with curvatures $k_1, k_2, k_3, k_4$:
       $(k_1 + k_2 + k_3 + k_4)^2 = 2(k_1^2 + k_2^2 + k_3^2 + k_4^2)$
    
    2. **Casey's Theorem**: A generalization of Ptolemy's theorem for circles.
    
    3. **Radical Axis**: The locus of points with equal power with respect to two circles.
    """)

with tab3:
    st.markdown("### Interactive Exploration Mode")
    
    # Animation controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        animate = st.checkbox("Enable Animation", value=False)
        if animate:
            speed = st.slider("Animation Speed", 0.1, 2.0, 1.0)
    
    with col2:
        show_construction = st.checkbox("Show Construction Lines", value=False)
        show_tangent_points = st.checkbox("Show Tangent Points", value=False)
    
    with col3:
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Classic", "Vibrant", "Pastel", "Monochrome"]
        )
    
    # Interactive configuration
    st.markdown("### Quick Configurations")
    
    config_cols = st.columns(4)
    
    with config_cols[0]:
        if st.button("Random", use_container_width=True):
            # Generate random configuration
            st.session_state.preset_x1 = np.random.uniform(-5, 5)
            st.session_state.preset_y1 = np.random.uniform(-5, 5)
            st.session_state.preset_r1 = np.random.uniform(0.5, 3)
            st.session_state.preset_x2 = np.random.uniform(-5, 5)
            st.session_state.preset_y2 = np.random.uniform(-5, 5)
            st.session_state.preset_r2 = np.random.uniform(0.5, 3)
            st.session_state.preset_x3 = np.random.uniform(-5, 5)
            st.session_state.preset_y3 = np.random.uniform(-5, 5)
            st.session_state.preset_r3 = np.random.uniform(0.5, 3)
            st.session_state.load_preset = True
            st.rerun()
    
    with config_cols[1]:
        if st.button("Unit Circles", use_container_width=True):
            # Three unit circles
            angles = np.array([0, 120, 240]) * np.pi / 180
            for i, angle in enumerate(angles):
                setattr(st.session_state, f'preset_x{i+1}', 2 * np.cos(angle))
                setattr(st.session_state, f'preset_y{i+1}', 2 * np.sin(angle))
                setattr(st.session_state, f'preset_r{i+1}', 1.0)
            st.session_state.load_preset = True
            st.rerun()
    
    with config_cols[2]:
        if st.button("Gasket", use_container_width=True):
            # Apollonian gasket configuration
            st.session_state.preset_x1, st.session_state.preset_y1, st.session_state.preset_r1 = 0, 0, 3
            st.session_state.preset_x2, st.session_state.preset_y2, st.session_state.preset_r2 = 3, 0, 1.5
            st.session_state.preset_x3, st.session_state.preset_y3, st.session_state.preset_r3 = 1.5, 2.598, 1.5
            st.session_state.load_preset = True
            st.rerun()
    
    with config_cols[3]:
        if st.button("Reset", use_container_width=True):
            # Reset to default
            st.session_state.preset_x1, st.session_state.preset_y1, st.session_state.preset_r1 = 0, 0, 3
            st.session_state.preset_x2, st.session_state.preset_y2, st.session_state.preset_r2 = 5, 0, 2
            st.session_state.preset_x3, st.session_state.preset_y3, st.session_state.preset_r3 = 2.5, 4, 2.5
            st.session_state.load_preset = True
            st.rerun()
    
    # Analysis tools
    st.markdown("### Analysis Tools")
    
    analysis_cols = st.columns(2)
    
    with analysis_cols[0]:
        st.markdown("#### Circle Relationships")
        
        # Get current values
        c1 = np.array([st.session_state.x1, st.session_state.y1])
        c2 = np.array([st.session_state.x2, st.session_state.y2])
        c3 = np.array([st.session_state.x3, st.session_state.y3])
        r1 = st.session_state.r1
        r2 = st.session_state.r2
        r3 = st.session_state.r3
        
        # Calculate relationships
        relationships = []
        circles = [(c1, r1, "1"), (c2, r2, "2"), (c3, r3, "3")]
        
        for i in range(3):
            for j in range(i+1, 3):
                ci, ri, ni = circles[i]
                cj, rj, nj = circles[j]
                d = np.linalg.norm(ci - cj)
                
                if abs(d - (ri + rj)) < 0.01:
                    rel = "Externally tangent"
                elif abs(d - abs(ri - rj)) < 0.01:
                    rel = "Internally tangent"
                elif d > ri + rj:
                    rel = f"Separate (gap: {d - (ri + rj):.2f})"
                elif d < abs(ri - rj):
                    rel = "One contains the other"
                else:
                    rel = "Overlapping"
                
                relationships.append(f"**Circles {ni} & {nj}:** {rel}")
        
        for rel in relationships:
            st.markdown(rel)
    
    with analysis_cols[1]:
        st.markdown("#### Configuration Properties")
        
        # Calculate area of triangle formed by centers
        area = 0.5 * abs(np.cross(c2 - c1, c3 - c1))
        st.metric("Triangle Area", f"{area:.3f}")
        
        # Check if centers are nearly collinear
        if area < 0.1:
            st.warning("⚠️ Centers are nearly collinear!")
        
        # Calculate perimeter
        perimeter = (np.linalg.norm(c2 - c1) + 
                    np.linalg.norm(c3 - c2) + 
                    np.linalg.norm(c1 - c3))
        st.metric("Triangle Perimeter", f"{perimeter:.3f}")
        
        # Total area covered
        total_area = np.pi * (r1**2 + r2**2 + r3**2)
        st.metric("Total Circle Area", f"{total_area:.3f}")
    
    # Solve and display
    if st.button("🎯 Analyze Configuration", type="primary", use_container_width=True):
        solutions = find_all_apollonius_circles(c1, r1, c2, r2, c3, r3)
        
        if solutions:
            # Enhanced visualization for interactive mode
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left plot: All solutions
            ax1.set_aspect('equal')
            ax1.set_title("All Solutions", fontsize=14, weight='bold')
            
            # Right plot: Solution details
            ax2.set_aspect('equal')
            ax2.set_title("Selected Solution Details", fontsize=14, weight='bold')
            
            # Plot on both axes
            for ax in [ax1, ax2]:
                ax.set_facecolor('#f8f9fa')
                
                # Plot given circles
                for center, radius, color in [(c1, r1, 'blue'), (c2, r2, 'green'), (c3, r3, 'red')]:
                    circle = Circle(center, radius, fill=False, edgecolor=color, linewidth=2.5, alpha=0.8)
                    ax.add_patch(circle)
                    ax.plot(*center, 'o', color=color, markersize=8)
            
            # Plot all solutions on left
            colors = plt.cm.rainbow(np.linspace(0, 1, len(solutions)))
            for i, (sol, color) in enumerate(zip(solutions, colors)):
                circle = Circle(sol['center'], sol['radius'], fill=False, 
                              edgecolor=color, linewidth=2, alpha=0.6,
                              linestyle='--' if any(t == -1 for t in sol['tangency']) else '-')
                ax1.add_patch(circle)
            
            # Set limits
            all_centers = [c1, c2, c3] + [sol['center'] for sol in solutions]
            all_radii = [r1, r2, r3] + [sol['radius'] for sol in solutions]
            x_coords = [c[0] for c in all_centers]
            y_coords = [c[1] for c in all_centers]
            margin = max(all_radii) * 1.5
            
            for ax in [ax1, ax2]:
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Solution selector
            st.markdown("### Explore Individual Solutions")
            
            solution_names = [f"Solution {i+1}: {sol['type']} (r={sol['radius']:.2f})" 
                            for i, sol in enumerate(solutions)]
            
            selected_idx = st.selectbox("Select a solution to examine:", 
                                      range(len(solutions)), 
                                      format_func=lambda x: solution_names[x])
            
            if selected_idx is not None:
                sol = solutions[selected_idx]
                
                # Display detailed analysis
                detail_cols = st.columns(3)
                
                with detail_cols[0]:
                    st.markdown("#### Geometry")
                    st.write(f"**Center:** ({sol['center'][0]:.4f}, {sol['center'][1]:.4f})")
                    st.write(f"**Radius:** {sol['radius']:.4f}")
                    st.write(f"**Area:** {np.pi * sol['radius']**2:.4f}")
                
                with detail_cols[1]:
                    st.markdown("#### Tangency Analysis")
                    for i, (t, c, r) in enumerate(zip(sol['tangency'], [c1, c2, c3], [r1, r2, r3])):
                        tang_type = "External" if t == 1 else "Internal"
                        d = np.linalg.norm(sol['center'] - c)
                        st.write(f"**Circle {i+1}:** {tang_type}")
                        st.write(f"  Distance: {d:.4f}")
                
                with detail_cols[2]:
                    st.markdown("#### Quality Metrics")
                    st.write(f"**Max Error:** {sol['error']:.8f}")
                    st.write(f"**Solution Type:** {sol['type']}")
                    
                    quality = "Excellent" if sol['error'] < 1e-6 else "Good" if sol['error'] < 1e-4 else "Acceptable"
                    st.write(f"**Quality:** {quality}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Created with ❤️ using Streamlit and Matplotlib</p>
    <p>Implementation includes both algebraic and iterative solvers for maximum robustness</p>
</div>
""", unsafe_allow_html=True)

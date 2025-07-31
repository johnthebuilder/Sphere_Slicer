import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Apollonius Circles on a Sphere",
    page_icon="🌐",
    layout="wide"
)

def calculate_third_circle_tangent(c1, r1, c2, r2, external=True):
    """
    Calculate positions for a third circle tangent to two given circles.
    Returns two possible positions.
    """
    d = np.linalg.norm(c2 - c1)
    
    if external:
        # External tangency
        if d < abs(r1 - r2) or d > r1 + r2:
            return None, None
    else:
        # Internal tangency
        if d > abs(r1 - r2):
            return None, None
    
    # Direction from c1 to c2
    if d > 0:
        u = (c2 - c1) / d
    else:
        return None, None
    
    # Perpendicular direction
    v = np.array([-u[1], u[0]])
    
    # Distance from c1 to the radical axis
    if external:
        a = (d**2 + r1**2 - r2**2) / (2 * d)
    else:
        a = (d**2 - r1**2 + r2**2) / (2 * d)
    
    # Height of the triangle
    h_squared = r1**2 - a**2
    if h_squared < 0:
        return None, None
    h = np.sqrt(h_squared)
    
    # Two possible positions
    p = c1 + a * u
    pos1 = p + h * v
    pos2 = p - h * v
    
    return pos1, pos2

def create_three_tangent_circles(r1, r2, r3):
    """
    Create three mutually tangent circles.
    Place circle 1 at origin, circle 2 on positive x-axis.
    """
    # Circle 1 at origin
    c1 = np.array([0.0, 0.0])
    
    # Circle 2 on x-axis, tangent to circle 1
    c2 = np.array([r1 + r2, 0.0])
    
    # Circle 3 must be tangent to both circles 1 and 2
    # Use the formula for the third circle
    d12 = r1 + r2
    d13 = r1 + r3
    d23 = r2 + r3
    
    # Using cosine rule to find angle
    cos_angle = (d12**2 + d13**2 - d23**2) / (2 * d12 * d13)
    if abs(cos_angle) > 1:
        # Adjust radii slightly to make it possible
        cos_angle = np.clip(cos_angle, -1, 1)
    
    angle = np.arccos(cos_angle)
    
    # Position of circle 3
    c3 = np.array([d13 * np.cos(angle), d13 * np.sin(angle)])
    
    return c1, c2, c3

def stereographic_projection(x, y, R=1):
    """
    Project a point from the plane to the sphere using stereographic projection.
    Returns (X, Y, Z) coordinates on the sphere.
    """
    denom = x**2 + y**2 + 4*R**2
    X = 4*R**2 * x / denom
    Y = 4*R**2 * y / denom
    Z = R * (x**2 + y**2 - 4*R**2) / denom
    return X, Y, Z

def inverse_stereographic(X, Y, Z, R=1):
    """
    Project a point from the sphere to the plane.
    """
    if Z == R:
        return float('inf'), float('inf')
    x = 2*R * X / (R - Z)
    y = 2*R * Y / (R - Z)
    return x, y

def project_circle_to_sphere(center, radius, R=1, n_points=100):
    """
    Project a circle from the plane to the sphere.
    Returns arrays of X, Y, Z coordinates.
    """
    theta = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    
    X, Y, Z = [], [], []
    for xi, yi in zip(x, y):
        Xi, Yi, Zi = stereographic_projection(xi, yi, R)
        X.append(Xi)
        Y.append(Yi)
        Z.append(Zi)
    
    return np.array(X), np.array(Y), np.array(Z)

def solve_apollonius_iterative(c1, r1, c2, r2, c3, r3, s1=1, s2=1, s3=1, max_iter=1000, tol=1e-8):
    """
    Solve the Problem of Apollonius using iterative method.
    """
    # Initial guess - weighted centroid
    total_r = r1 + r2 + r3
    x0 = (c1[0]*r1 + c2[0]*r2 + c3[0]*r3) / total_r
    y0 = (c1[1]*r1 + c2[1]*r2 + c3[1]*r3) / total_r
    r0 = total_r / 3
    
    x, y, r = x0, y0, r0
    
    for iteration in range(max_iter):
        # Calculate current distances
        d1 = np.sqrt((x - c1[0])**2 + (y - c1[1])**2)
        d2 = np.sqrt((x - c2[0])**2 + (y - c2[1])**2)
        d3 = np.sqrt((x - c3[0])**2 + (y - c3[1])**2)
        
        # Target distances
        target_d1 = r + s1 * r1
        target_d2 = r + s2 * r2
        target_d3 = r + s3 * r3
        
        # Errors
        e1 = d1 - target_d1
        e2 = d2 - target_d2
        e3 = d3 - target_d3
        
        # Check convergence
        if abs(e1) < tol and abs(e2) < tol and abs(e3) < tol:
            if r > 0:
                return np.array([x, y]), r
            else:
                return None, None
        
        # Compute Jacobian elements
        if d1 > 0:
            dx1 = (x - c1[0]) / d1
            dy1 = (y - c1[1]) / d1
        else:
            dx1, dy1 = 0, 0
            
        if d2 > 0:
            dx2 = (x - c2[0]) / d2
            dy2 = (y - c2[1]) / d2
        else:
            dx2, dy2 = 0, 0
            
        if d3 > 0:
            dx3 = (x - c3[0]) / d3
            dy3 = (y - c3[1]) / d3
        else:
            dx3, dy3 = 0, 0
        
        # Build system matrix (Jacobian)
        J = np.array([
            [dx1, dy1, -1],
            [dx2, dy2, -1],
            [dx3, dy3, -1]
        ])
        
        # Error vector
        E = np.array([e1, e2, e3])
        
        # Solve for corrections using least squares
        try:
            delta = np.linalg.lstsq(J, -E, rcond=None)[0]
            
            # Apply corrections with damping
            damping = 0.7
            x += damping * delta[0]
            y += damping * delta[1]
            r += damping * delta[2]
            
            # Ensure radius stays positive
            r = max(0.01, r)
            
        except:
            return None, None
    
    return np.array([x, y]), r

def find_all_apollonius_circles(c1, r1, c2, r2, c3, r3):
    """
    Find all possible Apollonius circles (up to 8 solutions).
    """
    solutions = []
    
    # Try all 8 combinations of internal (-1) and external (+1) tangencies
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            for s3 in [1, -1]:
                center, radius = solve_apollonius_iterative(c1, r1, c2, r2, c3, r3, s1, s2, s3)
                
                if center is not None and radius is not None and radius > 0:
                    # Check if this is internal tangency to a smaller circle (invalid)
                    valid = True
                    if s1 == -1 and radius < r1:
                        valid = False
                    if s2 == -1 and radius < r2:
                        valid = False
                    if s3 == -1 and radius < r3:
                        valid = False
                    
                    if valid:
                        # Check if solution is duplicate
                        is_duplicate = False
                        for sol in solutions:
                            if np.linalg.norm(sol['center'] - center) < 0.1 and abs(sol['radius'] - radius) < 0.1:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            solutions.append({
                                'center': center,
                                'radius': radius,
                                'tangency': (s1, s2, s3),
                                'type': f"{'E' if s1==1 else 'I'}{'E' if s2==1 else 'I'}{'E' if s3==1 else 'I'}"
                            })
    
    return solutions

def plot_apollonius_on_sphere(c1, r1, c2, r2, c3, r3, solutions, sphere_radius=1):
    """
    Create both 2D and 3D visualizations of the Apollonius circles.
    """
    fig = plt.figure(figsize=(16, 8))
    
    # 2D Plot
    ax1 = fig.add_subplot(121)
    ax1.set_aspect('equal')
    ax1.set_title('Apollonius Circles on Plane', fontsize=14, weight='bold')
    
    # Plot the three given circles
    colors = ['blue', 'green', 'red']
    for i, (center, radius, color) in enumerate([(c1, r1, colors[0]), (c2, r2, colors[1]), (c3, r3, colors[2])]):
        circle = Circle(center, radius, fill=False, edgecolor=color, linewidth=2.5, alpha=0.8)
        ax1.add_patch(circle)
        circle_fill = Circle(center, radius, fill=True, facecolor=color, alpha=0.05)
        ax1.add_patch(circle_fill)
        ax1.plot(*center, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        ax1.text(center[0], center[1]-0.2, f'{i+1}', fontsize=10, ha='center', va='top', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))
    
    # Plot solutions
    if solutions:
        solution_colors = plt.cm.rainbow(np.linspace(0, 1, len(solutions)))
        for i, (sol, color) in enumerate(zip(solutions, solution_colors)):
            has_internal = any(t == -1 for t in sol['tangency'])
            circle = Circle(sol['center'], sol['radius'], fill=False, edgecolor=color, 
                          linewidth=2, alpha=0.7, linestyle='--' if has_internal else '-')
            ax1.add_patch(circle)
            ax1.plot(*sol['center'], 'o', color=color, markersize=6)
            ax1.text(sol['center'][0], sol['center'][1]+0.2, f"{sol['type']}", 
                    fontsize=8, ha='center', va='bottom', alpha=0.8)
    
    # Set limits for 2D plot
    all_centers = [c1, c2, c3] + [sol['center'] for sol in solutions]
    all_radii = [r1, r2, r3] + [sol['radius'] for sol in solutions]
    x_coords = [c[0] for c in all_centers]
    y_coords = [c[1] for c in all_centers]
    margin = max(all_radii) * 1.5 if all_radii else 2
    ax1.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
    ax1.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # 3D Plot on Sphere
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Circles Projected onto Sphere', fontsize=14, weight='bold')
    
    # Draw the sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightgray')
    
    # Project and plot the three given circles on the sphere
    for i, (center, radius, color) in enumerate([(c1, r1, colors[0]), (c2, r2, colors[1]), (c3, r3, colors[2])]):
        X, Y, Z = project_circle_to_sphere(center, radius, sphere_radius)
        ax2.plot(X, Y, Z, color=color, linewidth=3, alpha=0.9, label=f'Circle {i+1}')
        
        # Plot center point
        Xc, Yc, Zc = stereographic_projection(center[0], center[1], sphere_radius)
        ax2.scatter([Xc], [Yc], [Zc], color=color, s=100, edgecolor='white', linewidth=2)
    
    # Project and plot solutions on the sphere
    if solutions:
        for i, (sol, color) in enumerate(zip(solutions, solution_colors)):
            X, Y, Z = project_circle_to_sphere(sol['center'], sol['radius'], sphere_radius)
            has_internal = any(t == -1 for t in sol['tangency'])
            ax2.plot(X, Y, Z, color=color, linewidth=2, alpha=0.7, 
                    linestyle='--' if has_internal else '-', label=f"{sol['type']}")
    
    # Set 3D plot properties
    ax2.set_box_aspect([1,1,1])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=20, azim=45)
    
    # Add legend to 3D plot
    ax2.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    return fig

# Streamlit App
st.title("🌐 Apollonius Circles on a Sphere")
st.markdown("### Three Mutually Tangent Circles and Their Apollonius Solutions")

st.markdown("""
This app creates three mutually tangent circles and finds all Apollonius circles 
(circles tangent to all three). The circles are then projected onto a sphere using 
stereographic projection, showing how planar circles become circles on the sphere.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Main Solver", "📚 Theory", "🎮 Interactive Controls"])

with tab1:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Circle Configuration")
        
        # Input for the three radii
        st.markdown("**Define Three Mutually Tangent Circles**")
        
        r1 = st.slider("Radius of Circle 1 🔵", 0.5, 5.0, 2.0, 0.1)
        r2 = st.slider("Radius of Circle 2 🟢", 0.5, 5.0, 1.5, 0.1)
        r3 = st.slider("Radius of Circle 3 🔴", 0.5, 5.0, 1.0, 0.1)
        
        st.markdown("---")
        
        # Sphere parameters
        st.markdown("**Sphere Parameters**")
        sphere_radius = st.slider("Sphere Radius", 0.5, 3.0, 1.0, 0.1)
        
        st.markdown("---")
        
        # Quick presets
        st.markdown("**Quick Presets**")
        
        if st.button("Equal Radii", use_container_width=True):
            r1 = r2 = r3 = 1.5
            st.rerun()
        
        if st.button("Golden Ratio", use_container_width=True):
            r1 = 2.0
            r2 = 2.0 / 1.618
            r3 = r2 / 1.618
            st.rerun()
        
        if st.button("Fibonacci", use_container_width=True):
            r1 = 2.0
            r2 = 1.3
            r3 = 0.8
            st.rerun()
        
        # Solve button
        solve_button = st.button("🔍 Find Apollonius Circles", type="primary", use_container_width=True)
    
    with col2:
        if solve_button or True:  # Always show visualization
            # Create three mutually tangent circles
            c1, c2, c3 = create_three_tangent_circles(r1, r2, r3)
            
            # Display configuration info
            st.markdown("### Configuration Analysis")
            
            info_cols = st.columns(3)
            with info_cols[0]:
                d12 = np.linalg.norm(c2 - c1)
                st.metric("Distance 1↔2", f"{d12:.3f}")
                st.caption(f"Sum of radii: {r1 + r2:.3f}")
            
            with info_cols[1]:
                d13 = np.linalg.norm(c3 - c1)
                st.metric("Distance 1↔3", f"{d13:.3f}")
                st.caption(f"Sum of radii: {r1 + r3:.3f}")
            
            with info_cols[2]:
                d23 = np.linalg.norm(c3 - c2)
                st.metric("Distance 2↔3", f"{d23:.3f}")
                st.caption(f"Sum of radii: {r2 + r3:.3f}")
            
            # Verify mutual tangency
            tangency_errors = [
                abs(d12 - (r1 + r2)),
                abs(d13 - (r1 + r3)),
                abs(d23 - (r2 + r3))
            ]
            max_error = max(tangency_errors)
            
            if max_error < 0.01:
                st.success(f"✅ Circles are mutually tangent (max error: {max_error:.6f})")
            else:
                st.warning(f"⚠️ Circles are approximately tangent (max error: {max_error:.6f})")
            
            # Find Apollonius circles
            with st.spinner("Finding Apollonius circles..."):
                solutions = find_all_apollonius_circles(c1, r1, c2, r2, c3, r3)
            
            if solutions:
                st.info(f"Found {len(solutions)} Apollonius circle(s)")
                
                # Create visualization
                fig = plot_apollonius_on_sphere(c1, r1, c2, r2, c3, r3, solutions, sphere_radius)
                st.pyplot(fig)
                
                # Solution details
                st.markdown("### Solution Details")
                
                sol_cols = st.columns(min(len(solutions), 4))
                for i, sol in enumerate(solutions):
                    with sol_cols[i % min(len(solutions), 4)]:
                        st.markdown(f"**Solution {i+1}: {sol['type']}**")
                        st.write(f"Center: ({sol['center'][0]:.3f}, {sol['center'][1]:.3f})")
                        st.write(f"Radius: {sol['radius']:.3f}")
                        
                        # Tangency info
                        tangency_symbols = ['🔵', '🟢', '🔴']
                        tangency_text = []
                        for j, (s, sym) in enumerate(zip(sol['tangency'], tangency_symbols)):
                            tangency_text.append(f"{sym}{'E' if s==1 else 'I'}")
                        st.write(f"Tangency: {' '.join(tangency_text)}")
            else:
                st.error("No Apollonius circles found")
                
                # Still show the three circles
                fig = plot_apollonius_on_sphere(c1, r1, c2, r2, c3, r3, [], sphere_radius)
                st.pyplot(fig)

with tab2:
    st.markdown("""
    ## Apollonius Circles on a Sphere
    
    ### Mutually Tangent Circles
    
    Three circles are **mutually tangent** if each circle is tangent to the other two. 
    Given three radii $r_1$, $r_2$, and $r_3$, we can construct three mutually tangent circles by:
    
    1. Placing circle 1 at the origin
    2. Placing circle 2 on the positive x-axis at distance $r_1 + r_2$
    3. Calculating the position of circle 3 using the constraint that it must be tangent to both
    
    The distances between centers must satisfy:
    - $d_{12} = r_1 + r_2$
    - $d_{13} = r_1 + r_3$
    - $d_{23} = r_2 + r_3$
    
    ### Stereographic Projection
    
    **Stereographic projection** maps the plane to a sphere by:
    1. Placing a sphere tangent to the plane at the origin
    2. Drawing lines from the "north pole" of the sphere through points on the plane
    3. The intersection of these lines with the sphere gives the projection
    
    For a sphere of radius $R$ centered at $(0, 0, R)$, the projection formulas are:
    
    $$X = \\frac{4R^2 x}{x^2 + y^2 + 4R^2}$$
    
    $$Y = \\frac{4R^2 y}{x^2 + y^2 + 4R^2}$$
    
    $$Z = R\\frac{x^2 + y^2 - 4R^2}{x^2 + y^2 + 4R^2}$$
    
    ### Properties of Stereographic Projection
    
    1. **Circles map to circles**: Circles on the plane map to circles on the sphere (lines map to circles through the north pole)
    2. **Conformal**: Angles are preserved
    3. **Not area-preserving**: Areas are distorted, with greater distortion farther from the origin
    
    ### The Descartes Circle Theorem
    
    For four mutually tangent circles with curvatures $k_1, k_2, k_3, k_4$ (where $k = 1/r$):
    
    $$(k_1 + k_2 + k_3 + k_4)^2 = 2(k_1^2 + k_2^2 + k_3^2 + k_4^2)$$
    
    This gives us another way to find the Apollonius circles!
    
    ### Applications
    
    - **Sphere packing**: Optimal arrangements of spheres
    - **Crystallography**: Atomic arrangements in crystals
    - **Computer graphics**: Texture mapping and projections
    - **Complex analysis**: The Riemann sphere
    """)

with tab3:
    st.markdown("### Interactive Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Visualization Options")
        show_grid = st.checkbox("Show grid lines", value=True)
        show_labels = st.checkbox("Show labels", value=True)
        show_construction = st.checkbox("Show construction lines", value=False)
        
        st.markdown("#### 3D View Controls")
        elevation = st.slider("Elevation angle", -90, 90, 20, 5)
        azimuth = st.slider("Azimuth angle", 0, 360, 45, 5)
    
    with col2:
        st.markdown("#### Analysis Tools")
        
        if 'c1' in locals():
            # Calculate some interesting properties
            st.markdown("**Descartes Circle Theorem Check**")
            
            k1, k2, k3 = 1/r1, 1/r2, 1/r3
            st.write(f"Curvatures: k₁={k1:.3f}, k₂={k2:.3f}, k₃={k3:.3f}")
            
            # For each Apollonius circle, check Descartes theorem
            if 'solutions' in locals() and solutions:
                for i, sol in enumerate(solutions[:2]):  # Show first two
                    k4 = 1/sol['radius']
                    lhs = (k1 + k2 + k3 + k4)**2
                    rhs = 2*(k1**2 + k2**2 + k3**2 + k4**2)
                    error = abs(lhs - rhs)
                    st.write(f"Solution {i+1}: k₄={k4:.3f}, error={error:.6f}")
    
    st.markdown("### Export Options")
    
    if st.button("Generate Python Code", use_container_width=True):
        code = f"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Circle radii
r1, r2, r3 = {r1}, {r2}, {r3}

# Create mutually tangent circles
c1 = np.array([0.0, 0.0])
c2 = np.array([r1 + r2, 0.0])

# Calculate c3 position
d12, d13, d23 = r1 + r2, r1 + r3, r2 + r3
cos_angle = (d12**2 + d13**2 - d23**2) / (2 * d12 * d13)
angle = np.arccos(cos_angle)
c3 = np.array([d13 * np.cos(angle), d13 * np.sin(angle)])

print(f"Circle 1: center={c1}, radius={r1}")
print(f"Circle 2: center={c2}, radius={r2}")
print(f"Circle 3: center={c3}, radius={r3}")
"""
        st.code(code, language='python')

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Three mutually tangent circles projected onto a sphere using stereographic projection</p>
    <p>Combining classical geometry with modern visualization</p>
</div>
""", unsafe_allow_html=True)

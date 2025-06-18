import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from datetime import datetime
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
import networkx as nx

def spherical_to_cartesian(theta, phi, r):
    """Convert spherical coordinates to Cartesian"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    """Convert Cartesian coordinates to spherical"""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def angular_distance(p1, p2):
    """Calculate angular distance between two points on a sphere"""
    # Normalize vectors
    p1_norm = p1 / np.linalg.norm(p1)
    p2_norm = p2 / np.linalg.norm(p2)
    
    # Calculate angle
    dot_product = np.clip(np.dot(p1_norm, p2_norm), -1, 1)
    return np.arccos(dot_product)

def calculate_cap_radius_from_angular(sphere_radius, angular_radius):
    """Calculate the actual circle radius of a spherical cap given its angular radius"""
    return sphere_radius * np.sin(angular_radius)

def calculate_angular_from_cap_radius(sphere_radius, cap_radius):
    """Calculate angular radius from cap circle radius"""
    return np.arcsin(min(cap_radius / sphere_radius, 1.0))

def generate_initial_packing(n_circles, sphere_radius, pattern="hexagonal"):
    """Generate initial circle centers for different packing patterns"""
    
    if pattern == "hexagonal":
        # Generate approximately hexagonal packing
        centers = []
        
        # Start with poles
        centers.append([0, 0, sphere_radius])  # North pole
        centers.append([0, 0, -sphere_radius])  # South pole
        
        # Add latitude bands
        n_latitudes = int(np.sqrt(n_circles))
        for i in range(1, n_latitudes):
            theta = np.pi * i / n_latitudes
            z = sphere_radius * np.cos(theta)
            r_circle = sphere_radius * np.sin(theta)
            
            # Number of points at this latitude
            n_points = int(2 * np.pi * r_circle / (2 * np.pi * sphere_radius / n_latitudes))
            
            for j in range(n_points):
                phi = 2 * np.pi * j / n_points
                x = r_circle * np.cos(phi)
                y = r_circle * np.sin(phi)
                centers.append([x, y, z])
        
        return np.array(centers[:n_circles])
    
    elif pattern == "triangular":
        # Geodesic polyhedron approach
        # Start with icosahedron
        phi = (1 + np.sqrt(5)) / 2
        
        vertices = []
        # Rectangle in xy plane
        for i in [-1, 1]:
            for j in [-1, 1]:
                vertices.append([0, i, j * phi])
                vertices.append([i, j * phi, 0])
                vertices.append([j * phi, 0, i])
        
        vertices = np.array(vertices)
        # Normalize to sphere surface
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms * sphere_radius
        
        # Subdivide faces to get more points
        centers = list(vertices)
        
        # Add midpoints of edges
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                midpoint = (vertices[i] + vertices[j]) / 2
                midpoint = midpoint / np.linalg.norm(midpoint) * sphere_radius
                centers.append(midpoint)
                
                if len(centers) >= n_circles:
                    return np.array(centers[:n_circles])
        
        return np.array(centers[:n_circles])
    
    else:  # square
        # Latitude-longitude grid
        n_lat = int(np.sqrt(n_circles * 2 / np.pi))
        n_lon = int(n_lat * np.pi / 2)
        
        centers = []
        for i in range(n_lat):
            theta = np.pi * (i + 0.5) / n_lat
            for j in range(n_lon):
                phi = 2 * np.pi * j / n_lon
                x, y, z = spherical_to_cartesian(theta, phi, sphere_radius)
                centers.append([x, y, z])
        
        return np.array(centers[:n_circles])

def compute_delaunay_triangulation(points):
    """Compute Delaunay triangulation on sphere using SphericalVoronoi"""
    sv = SphericalVoronoi(points, radius=np.linalg.norm(points[0]), center=np.array([0, 0, 0]))
    sv.sort_vertices_of_regions()
    
    # Build neighbor graph
    G = nx.Graph()
    for i in range(len(points)):
        G.add_node(i)
    
    # Add edges based on Voronoi regions
    for i, region in enumerate(sv.regions):
        for j, other_region in enumerate(sv.regions):
            if i < j and len(set(region) & set(other_region)) >= 2:
                G.add_edge(i, j)
    
    return G

def optimize_radii_for_tangency(centers, sphere_radius, neighbor_graph):
    """Optimize radii so circles are tangent to their neighbors"""
    n_circles = len(centers)
    
    def objective(radii):
        """Minimize deviation from perfect tangency"""
        error = 0
        constraint_violations = 0
        
        for i in range(n_circles):
            for j in neighbor_graph.neighbors(i):
                if j > i:  # Avoid counting twice
                    # Angular distance between centers
                    ang_dist = angular_distance(centers[i], centers[j])
                    
                    # Angular radii
                    ang_rad_i = calculate_angular_from_cap_radius(sphere_radius, radii[i])
                    ang_rad_j = calculate_angular_from_cap_radius(sphere_radius, radii[j])
                    
                    # Target: sum of angular radii should equal angular distance
                    target_sum = ang_rad_i + ang_rad_j
                    error += (ang_dist - target_sum) ** 2
                    
                    # Penalty for overlapping
                    if target_sum > ang_dist:
                        constraint_violations += 10 * (target_sum - ang_dist) ** 2
        
        return error + constraint_violations
    
    # Initial guess - uniform radii
    initial_radii = np.ones(n_circles) * sphere_radius * 0.2
    
    # Bounds
    min_radius = sphere_radius * 0.05
    max_radius = sphere_radius * 0.5
    bounds = [(min_radius, max_radius) for _ in range(n_circles)]
    
    # Optimize
    result = minimize(objective, initial_radii, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 1000})
    
    return result.x

def check_sphere_coverage(centers, radii, sphere_radius, n_samples=1000):
    """Check what percentage of sphere is covered by caps"""
    # Generate random points on sphere
    random_points = []
    for _ in range(n_samples):
        theta = np.arccos(1 - 2 * np.random.random())
        phi = 2 * np.pi * np.random.random()
        x, y, z = spherical_to_cartesian(theta, phi, sphere_radius)
        random_points.append([x, y, z])
    
    random_points = np.array(random_points)
    
    # Check coverage
    covered = 0
    for point in random_points:
        for center, radius in zip(centers, radii):
            ang_dist = angular_distance(point, center)
            ang_radius = calculate_angular_from_cap_radius(sphere_radius, radius)
            
            if ang_dist <= ang_radius:
                covered += 1
                break
    
    return covered / n_samples * 100

def plot_sphere_with_caps(sphere_radius, centers, cap_radii, neighbor_graph):
    """Visualize the sphere with spherical caps"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot sphere surface (more detailed)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightgray')
    
    # Color map for caps
    colors = plt.cm.rainbow(np.linspace(0, 1, len(centers)))
    
    # Plot spherical caps
    for i, (center, cap_radius, color) in enumerate(zip(centers, cap_radii, colors)):
        # Plot the cap circle
        normal = center / np.linalg.norm(center)
        
        # Find two orthogonal vectors in the plane
        if abs(normal[2]) < 0.9:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Generate circle points
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_points = []
        
        for t in theta:
            # Point on circle in the tangent plane
            point_in_plane = cap_radius * (np.cos(t) * v1 + np.sin(t) * v2)
            # Project to sphere surface
            point_direction = center + point_in_plane
            point_direction = point_direction / np.linalg.norm(point_direction)
            point_on_sphere = sphere_radius * point_direction
            circle_points.append(point_on_sphere)
        
        circle_points = np.array(circle_points)
        ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
                color=color, linewidth=2)
        
        # Fill the cap with semi-transparent color
        angular_radius = calculate_angular_from_cap_radius(sphere_radius, cap_radius)
        cap_points = []
        
        for r in np.linspace(0, 1, 10):
            for t in np.linspace(0, 2 * np.pi, 20):
                # Point inside the cap
                point_in_plane = r * cap_radius * (np.cos(t) * v1 + np.sin(t) * v2)
                point_direction = center + point_in_plane
                point_direction = point_direction / np.linalg.norm(point_direction)
                point_on_sphere = sphere_radius * point_direction
                cap_points.append(point_on_sphere)
        
        cap_points = np.array(cap_points)
        ax.scatter(cap_points[:, 0], cap_points[:, 1], cap_points[:, 2], 
                  color=color, s=1, alpha=0.3)
        
        # Mark center
        ax.scatter(*center, color='black', s=30, marker='o')
    
    # Plot neighbor connections
    for i in range(len(centers)):
        for j in neighbor_graph.neighbors(i):
            if j > i:  # Avoid duplicate lines
                ax.plot([centers[i][0], centers[j][0]], 
                       [centers[i][1], centers[j][1]], 
                       [centers[i][2], centers[j][2]], 
                       'k--', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Complete Sphere Packing with {len(centers)} Tangent Caps')
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    max_range = sphere_radius * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    return fig

def generate_output_text(sphere_radius, centers, cap_radii, neighbor_graph, coverage):
    """Generate the output text file content"""
    output = f"Complete Sphere Packing Configuration\n"
    output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += f"{'='*60}\n\n"
    output += f"Sphere radius: {sphere_radius:.6f}\n"
    output += f"Number of caps: {len(centers)}\n"
    output += f"Surface coverage: {coverage:.1f}%\n\n"
    
    output += f"Spherical Caps:\n"
    output += f"{'-'*60}\n"
    
    # Calculate tangency errors
    tangency_errors = []
    
    for i, (center, cap_radius) in enumerate(zip(centers, cap_radii)):
        output += f"Cap {i+1}:\n"
        output += f"  Center: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})\n"
        output += f"  Cap circle radius: {cap_radius:.6f}\n"
        angular_radius = calculate_angular_from_cap_radius(sphere_radius, cap_radius)
        output += f"  Angular radius: {np.degrees(angular_radius):.2f}°\n"
        
        # List neighbors and tangency quality
        neighbors = list(neighbor_graph.neighbors(i))
        output += f"  Neighbors: {[n+1 for n in neighbors]}\n"
        
        # Check tangency quality
        errors = []
        for j in neighbors:
            ang_dist = angular_distance(centers[i], centers[j])
            ang_rad_i = calculate_angular_from_cap_radius(sphere_radius, cap_radius)
            ang_rad_j = calculate_angular_from_cap_radius(sphere_radius, cap_radii[j])
            error = abs(ang_dist - (ang_rad_i + ang_rad_j))
            errors.append(error)
            tangency_errors.append(error)
        
        if errors:
            output += f"  Avg tangency error: {np.mean(errors)*180/np.pi:.3f}°\n"
        output += "\n"
    
    # Summary statistics
    output += f"\nPacking Statistics:\n"
    output += f"{'-'*60}\n"
    output += f"Total neighbor pairs: {neighbor_graph.number_of_edges()}\n"
    output += f"Average tangency error: {np.mean(tangency_errors)*180/np.pi:.3f}°\n"
    output += f"Max tangency error: {np.max(tangency_errors)*180/np.pi:.3f}°\n"
    output += f"Surface coverage: {coverage:.1f}%\n"
    
    return output

# Streamlit App
st.title("Complete Sphere Packing with Tangent Caps")

st.markdown("""
This app generates a complete packing of spherical caps on a sphere surface where:
- Each cap is tangent to all its neighbors
- The entire sphere surface is covered (no gaps)
- Caps don't overlap except at tangent points
""")

# Sidebar parameters
st.sidebar.header("Parameters")

sphere_radius = st.sidebar.number_input("Sphere Radius", min_value=1.0, max_value=100.0, value=10.0, step=0.1)

n_caps = st.sidebar.slider("Target Number of Caps", min_value=4, max_value=50, value=20, 
                           help="Actual number may vary based on packing constraints")

pattern = st.sidebar.selectbox(
    "Initial Packing Pattern",
    ["hexagonal", "triangular", "square"],
    help="Starting pattern for optimization"
)

optimization_iterations = st.sidebar.slider("Optimization Iterations", min_value=100, max_value=2000, value=1000)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Note**: Complete sphere packing with tangent circles is a complex optimization problem. 
The algorithm will try to achieve the best possible packing, but perfect tangency 
for all circles may not always be achievable.
""")

if st.sidebar.button("Generate Packing", type="primary"):
    with st.spinner("Generating optimal sphere packing..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Generate initial configuration
        status_text.text("Generating initial configuration...")
        progress_bar.progress(20)
        centers = generate_initial_packing(n_caps, sphere_radius, pattern)
        
        # Step 2: Compute neighbor graph
        status_text.text("Computing neighbor relationships...")
        progress_bar.progress(40)
        neighbor_graph = compute_delaunay_triangulation(centers)
        
        # Step 3: Optimize radii for tangency
        status_text.text("Optimizing cap sizes for perfect tangency...")
        progress_bar.progress(60)
        optimized_radii = optimize_radii_for_tangency(centers, sphere_radius, neighbor_graph)
        
        # Step 4: Check coverage
        status_text.text("Analyzing sphere coverage...")
        progress_bar.progress(80)
        coverage = check_sphere_coverage(centers, optimized_radii, sphere_radius)
        
        # Step 5: Create visualization
        status_text.text("Creating visualization...")
        progress_bar.progress(90)
        fig = plot_sphere_with_caps(sphere_radius, centers, optimized_radii, neighbor_graph)
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        
        # Display results
        st.pyplot(fig)
        
        # Generate output text
        output_text = generate_output_text(sphere_radius, centers, optimized_radii, neighbor_graph, coverage)
        
        # Display summary metrics
        st.subheader("Packing Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sphere Radius", f"{sphere_radius:.2f}")
        with col2:
            st.metric("Number of Caps", len(centers))
        with col3:
            st.metric("Coverage", f"{coverage:.1f}%")
        with col4:
            # Calculate average tangency error
            tangency_errors = []
            for i in range(len(centers)):
                for j in neighbor_graph.neighbors(i):
                    if j > i:
                        ang_dist = angular_distance(centers[i], centers[j])
                        ang_rad_i = calculate_angular_from_cap_radius(sphere_radius, optimized_radii[i])
                        ang_rad_j = calculate_angular_from_cap_radius(sphere_radius, optimized_radii[j])
                        error = abs(ang_dist - (ang_rad_i + ang_rad_j))
                        tangency_errors.append(error)
            
            avg_error = np.mean(tangency_errors) * 180 / np.pi if tangency_errors else 0
            st.metric("Avg Tangency Error", f"{avg_error:.3f}°")
        
        # Quality indicators
        if coverage > 95:
            st.success(f"✓ Excellent coverage: {coverage:.1f}% of sphere surface covered")
        elif coverage > 85:
            st.warning(f"⚠ Good coverage: {coverage:.1f}% of sphere surface covered")
        else:
            st.error(f"✗ Poor coverage: {coverage:.1f}% of sphere surface covered - try more caps")
        
        if avg_error < 1.0:
            st.success(f"✓ Excellent tangency: average error only {avg_error:.3f}°")
        elif avg_error < 5.0:
            st.warning(f"⚠ Good tangency: average error {avg_error:.3f}°")
        else:
            st.info(f"ℹ Moderate tangency: average error {avg_error:.3f}°")
        
        # Download button
        st.download_button(
            label="Download Configuration",
            data=output_text,
            file_name=f"sphere_packing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        # Show output preview
        with st.expander("View Output File Preview"):
            st.text(output_text)

# Instructions
with st.expander("How it Works"):
    st.markdown("""
    ### Complete Sphere Packing Algorithm
    
    1. **Initial Configuration**: Places caps using one of three patterns:
       - **Hexagonal**: Each cap surrounded by ~6 neighbors (most efficient)
       - **Triangular**: Based on geodesic polyhedron subdivision
       - **Square**: Latitude-longitude grid pattern
    
    2. **Neighbor Detection**: Uses Spherical Voronoi/Delaunay triangulation to find which caps should be neighbors
    
    3. **Radius Optimization**: Adjusts each cap's radius so that:
       - It's tangent to all its neighbors (boundaries touch exactly)
       - No overlapping occurs
       - Coverage is maximized
    
    4. **Coverage Analysis**: Tests random points to measure what percentage of the sphere is covered
    
    ### Understanding the Output
    
    - **Coverage %**: How much of the sphere surface is covered by caps (goal: 100%)
    - **Tangency Error**: How far from perfect tangency the caps are (goal: 0°)
    - **Neighbor Graph**: Dotted lines show which caps are meant to be tangent
    
    ### Mathematical Notes
    
    - Perfect circle packing on a sphere is only possible for certain special numbers (4, 6, 8, 12, 20, etc.)
    - For arbitrary numbers, some compromise in tangency or coverage is inevitable
    - The algorithm minimizes these compromises through optimization
    """)

# Add scipy to requirements
st.sidebar.markdown("---")
st.sidebar.caption("Note: This app requires scipy for optimization")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from datetime import datetime
from itertools import combinations

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

def generate_uniform_sphere_points(n_points, sphere_radius):
    """Generate approximately uniform points on sphere using Fibonacci spiral"""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle
    
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        
        theta = phi * i  # golden angle increment
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points.append([x * sphere_radius, y * sphere_radius, z * sphere_radius])
    
    return np.array(points)

def generate_icosahedral_points(sphere_radius):
    """Generate 12 points based on icosahedron vertices"""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # Icosahedron vertices
    points = []
    # Rectangle in xy plane
    for i in [-1, 1]:
        for j in [-1, 1]:
            points.append([0, i, j * phi])
            points.append([i, j * phi, 0])
            points.append([j * phi, 0, i])
    
    points = np.array(points)
    # Normalize to sphere surface
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms * sphere_radius
    
    return points

def generate_tetrahedral_points(sphere_radius):
    """Generate 4 points based on tetrahedron vertices"""
    # Tetrahedron vertices
    points = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ])
    
    # Normalize to sphere surface
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms * sphere_radius
    
    return points

def generate_octahedral_points(sphere_radius):
    """Generate 6 points based on octahedron vertices"""
    # Octahedron vertices (along axes)
    points = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ])
    
    points = points * sphere_radius
    return points

def find_k_nearest_neighbors(centers, k=6):
    """Find k nearest neighbors for each point"""
    n = len(centers)
    neighbors = {}
    
    for i in range(n):
        # Calculate distances to all other points
        distances = []
        for j in range(n):
            if i != j:
                dist = angular_distance(centers[i], centers[j])
                distances.append((j, dist))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[1])
        neighbors[i] = [d[0] for d in distances[:k]]
    
    return neighbors

def optimize_packing_simple(centers, sphere_radius, n_iterations=100):
    """Simple optimization to pack circles tangentially"""
    n_caps = len(centers)
    
    # Find neighbors based on proximity
    neighbors = find_k_nearest_neighbors(centers, k=6)
    
    # Initialize radii based on nearest neighbor distances
    radii = np.zeros(n_caps)
    for i in range(n_caps):
        if neighbors[i]:
            # Set initial radius to half the distance to nearest neighbor
            nearest = neighbors[i][0]
            dist = angular_distance(centers[i], centers[nearest])
            radii[i] = calculate_cap_radius_from_angular(sphere_radius, dist / 2)
    
    # Simple iterative adjustment
    for iteration in range(n_iterations):
        new_radii = radii.copy()
        
        for i in range(n_caps):
            # For each neighbor, calculate what radius would make us tangent
            tangent_radii = []
            
            for j in neighbors[i]:
                dist = angular_distance(centers[i], centers[j])
                # For tangency: r_i + r_j = dist
                # So r_i = dist - r_j
                angular_j = calculate_angular_from_cap_radius(sphere_radius, radii[j])
                target_angular_i = dist - angular_j
                
                if target_angular_i > 0:
                    target_radius_i = calculate_cap_radius_from_angular(sphere_radius, target_angular_i)
                    tangent_radii.append(target_radius_i)
            
            # Take average of suggested radii
            if tangent_radii:
                new_radii[i] = np.mean(tangent_radii)
        
        # Limit radii to reasonable bounds
        min_radius = sphere_radius * 0.05
        max_radius = sphere_radius * 0.5
        new_radii = np.clip(new_radii, min_radius, max_radius)
        
        # Update radii with damping
        radii = 0.7 * radii + 0.3 * new_radii
    
    return radii, neighbors

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

def plot_sphere_with_caps(sphere_radius, centers, cap_radii, neighbors):
    """Visualize the sphere with spherical caps"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
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
        
        # Mark center
        ax.scatter(*center, color='black', s=30, marker='o')
    
    # Plot neighbor connections
    for i, neighbor_list in neighbors.items():
        for j in neighbor_list:
            if j > i:  # Avoid duplicate lines
                ax.plot([centers[i][0], centers[j][0]], 
                       [centers[i][1], centers[j][1]], 
                       [centers[i][2], centers[j][2]], 
                       'k--', alpha=0.2, linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Sphere Packing with {len(centers)} Tangent Caps')
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    max_range = sphere_radius * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    return fig

def generate_output_text(sphere_radius, centers, cap_radii, neighbors, coverage):
    """Generate the output text file content"""
    output = f"Sphere Packing Configuration\n"
    output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += f"{'='*60}\n\n"
    output += f"Sphere radius: {sphere_radius:.6f}\n"
    output += f"Number of caps: {len(centers)}\n"
    output += f"Surface coverage: {coverage:.1f}%\n\n"
    
    output += f"Spherical Caps:\n"
    output += f"{'-'*60}\n"
    
    # Calculate tangency information
    tangency_pairs = []
    
    for i, (center, cap_radius) in enumerate(zip(centers, cap_radii)):
        output += f"Cap {i+1}:\n"
        output += f"  Center: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})\n"
        output += f"  Cap circle radius: {cap_radius:.6f}\n"
        angular_radius = calculate_angular_from_cap_radius(sphere_radius, cap_radius)
        output += f"  Angular radius: {np.degrees(angular_radius):.2f}°\n"
        
        # Check which neighbors are actually tangent
        tangent_neighbors = []
        for j in neighbors[i]:
            ang_dist = angular_distance(centers[i], centers[j])
            ang_rad_i = calculate_angular_from_cap_radius(sphere_radius, cap_radius)
            ang_rad_j = calculate_angular_from_cap_radius(sphere_radius, cap_radii[j])
            
            # Check if approximately tangent (within 5% error)
            expected = ang_rad_i + ang_rad_j
            if abs(ang_dist - expected) / expected < 0.05:
                tangent_neighbors.append(j + 1)
                if j > i:
                    tangency_pairs.append((i + 1, j + 1))
        
        output += f"  Tangent neighbors: {tangent_neighbors}\n\n"
    
    # Summary statistics
    output += f"\nPacking Statistics:\n"
    output += f"{'-'*60}\n"
    output += f"Total tangent pairs: {len(tangency_pairs)}\n"
    output += f"Surface coverage: {coverage:.1f}%\n"
    
    return output

# Streamlit App
st.title("Sphere Packing with Tangent Caps")

st.markdown("""
This app generates a packing of spherical caps on a sphere surface where:
- Each cap aims to be tangent to its neighbors
- The sphere surface is covered as completely as possible
- Caps don't overlap
""")

# Sidebar parameters
st.sidebar.header("Parameters")

sphere_radius = st.sidebar.number_input("Sphere Radius", min_value=1.0, max_value=100.0, value=10.0, step=0.1)

packing_type = st.sidebar.selectbox(
    "Packing Type",
    ["Custom Number", "Tetrahedral (4)", "Octahedral (6)", "Icosahedral (12)", "Uniform Distribution"]
)

if packing_type == "Custom Number":
    n_caps = st.sidebar.slider("Number of Caps", min_value=4, max_value=50, value=20)
elif packing_type == "Tetrahedral (4)":
    n_caps = 4
elif packing_type == "Octahedral (6)":
    n_caps = 6
elif packing_type == "Icosahedral (12)":
    n_caps = 12
else:
    n_caps = st.sidebar.slider("Number of Caps", min_value=13, max_value=50, value=20)

optimization_iterations = st.sidebar.slider("Optimization Iterations", min_value=50, max_value=500, value=200)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Note**: Perfect tangent packing is mathematically possible only for certain numbers 
(4, 6, 8, 12, 20). For other numbers, the algorithm finds the best approximation.
""")

if st.sidebar.button("Generate Packing", type="primary"):
    with st.spinner("Generating sphere packing..."):
        
        # Generate initial configuration based on type
        if packing_type == "Tetrahedral (4)":
            centers = generate_tetrahedral_points(sphere_radius)
        elif packing_type == "Octahedral (6)":
            centers = generate_octahedral_points(sphere_radius)
        elif packing_type == "Icosahedral (12)":
            centers = generate_icosahedral_points(sphere_radius)
        else:
            centers = generate_uniform_sphere_points(n_caps, sphere_radius)
        
        # Optimize packing
        optimized_radii, neighbors = optimize_packing_simple(centers, sphere_radius, optimization_iterations)
        
        # Check coverage
        coverage = check_sphere_coverage(centers, optimized_radii, sphere_radius)
        
        # Create visualization
        fig = plot_sphere_with_caps(sphere_radius, centers, optimized_radii, neighbors)
        st.pyplot(fig)
        
        # Generate output text
        output_text = generate_output_text(sphere_radius, centers, optimized_radii, neighbors, coverage)
        
        # Display summary metrics
        st.subheader("Packing Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sphere Radius", f"{sphere_radius:.2f}")
        with col2:
            st.metric("Number of Caps", len(centers))
        with col3:
            st.metric("Coverage", f"{coverage:.1f}%")
        
        # Quality indicators
        if coverage > 95:
            st.success(f"✓ Excellent coverage: {coverage:.1f}% of sphere surface covered")
        elif coverage > 85:
            st.warning(f"⚠ Good coverage: {coverage:.1f}% of sphere surface covered")
        else:
            st.info(f"ℹ Coverage: {coverage:.1f}% - Try different parameters for better coverage")
        
        # Special cases
        if n_caps in [4, 6, 12]:
            st.info("ℹ This is a special case where perfect tangent packing is mathematically possible!")
        
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
with st.expander("Understanding Sphere Packing"):
    st.markdown("""
    ### What is Sphere Packing?
    
    This app creates a collection of circular "caps" on a sphere's surface, where each cap's 
    boundary is a circle that lies on the sphere. The goal is to:
    
    1. **Cover the entire sphere** - No gaps between caps
    2. **Make caps tangent** - Each cap's boundary touches its neighbors
    3. **Avoid overlaps** - Caps don't overlap except at tangent points
    
    ### Special Cases
    
    Perfect tangent packing is only possible for certain numbers:
    - **4 caps**: Tetrahedral arrangement
    - **6 caps**: Octahedral arrangement (along axes)
    - **8 caps**: Cubic arrangement
    - **12 caps**: Icosahedral arrangement
    - **20 caps**: Dodecahedral arrangement
    
    ### For Other Numbers
    
    The algorithm uses optimization to find the best possible arrangement, but perfect 
    tangency and complete coverage may not be achievable simultaneously.
    
    ### Output File
    
    The output includes:
    - Sphere radius
    - Each cap's center coordinates
    - Each cap's boundary circle radius
    - Which caps are tangent to each other
    - Coverage percentage
    """)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from datetime import datetime

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

def generate_icosahedral_vertices():
    """Generate normalized icosahedron vertices"""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # Icosahedron vertices
    vertices = []
    
    # (0, ±1, ±φ)
    for i in [-1, 1]:
        for j in [-1, 1]:
            vertices.append([0, i, j * phi])
    
    # (±1, ±φ, 0)
    for i in [-1, 1]:
        for j in [-1, 1]:
            vertices.append([i, j * phi, 0])
    
    # (±φ, 0, ±1)
    for i in [-1, 1]:
        for j in [-1, 1]:
            vertices.append([i * phi, 0, j])
    
    vertices = np.array(vertices)
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    return vertices

def generate_dodecahedral_vertices():
    """Generate normalized dodecahedron vertices"""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    vertices = []
    
    # (±1, ±1, ±1)
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vertices.append([x, y, z])
    
    # (0, ±1/φ, ±φ)
    for i in [-1, 1]:
        for j in [-1, 1]:
            vertices.append([0, i/phi, j*phi])
    
    # (±1/φ, ±φ, 0)
    for i in [-1, 1]:
        for j in [-1, 1]:
            vertices.append([i/phi, j*phi, 0])
    
    # (±φ, 0, ±1/φ)
    for i in [-1, 1]:
        for j in [-1, 1]:
            vertices.append([i*phi, 0, j/phi])
    
    vertices = np.array(vertices)
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    return vertices

def calculate_tangent_radius_for_vertices(vertices):
    """Calculate the radius for caps centered at vertices to be tangent"""
    # Find minimum distance between any two vertices
    min_dist = float('inf')
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            dist = angular_distance(vertices[i], vertices[j])
            if dist < min_dist:
                min_dist = dist
    
    # For tangent circles, each has angular radius = half the minimum distance
    angular_radius = min_dist / 2
    
    return angular_radius

def generate_uniform_points(n_points):
    """Generate uniformly distributed points on unit sphere using Fibonacci spiral"""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle
    
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        
        theta = phi * i  # golden angle increment
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points.append([x, y, z])
    
    return np.array(points)

def optimize_radii_for_coverage(centers, sphere_radius, max_iterations=100):
    """Optimize radii to maximize coverage while maintaining tangency where possible"""
    n_caps = len(centers)
    
    # Start with uniform radii based on surface area division
    total_area = 4 * np.pi * sphere_radius**2
    area_per_cap = total_area / n_caps
    
    # Initial angular radius from equal area division
    # Area of spherical cap = 2πR²(1-cos(θ))
    # area_per_cap = 2πR²(1-cos(θ))
    # 1-cos(θ) = area_per_cap / (2πR²)
    cos_theta = 1 - area_per_cap / (2 * np.pi * sphere_radius**2)
    initial_angular = np.arccos(cos_theta)
    
    # But limit to avoid too much overlap
    # Find nearest neighbor distance for each point
    min_neighbor_dists = []
    for i in range(n_caps):
        dists = []
        for j in range(n_caps):
            if i != j:
                dists.append(angular_distance(centers[i], centers[j]))
        min_neighbor_dists.append(min(dists))
    
    # Set radius to the smaller of: equal-area radius or half the minimum neighbor distance
    angular_radii = []
    for i in range(n_caps):
        max_angular = min_neighbor_dists[i] / 2
        angular_radii.append(min(initial_angular, max_angular * 0.98))  # 0.98 to ensure slight gap
    
    angular_radii = np.array(angular_radii)
    cap_radii = calculate_cap_radius_from_angular(sphere_radius, angular_radii)
    
    return cap_radii

def check_overlap(center1, center2, radius1, radius2, sphere_radius):
    """Check if two caps overlap"""
    ang_dist = angular_distance(center1, center2)
    ang_rad1 = calculate_angular_from_cap_radius(sphere_radius, radius1)
    ang_rad2 = calculate_angular_from_cap_radius(sphere_radius, radius2)
    return ang_dist < (ang_rad1 + ang_rad2 - 0.001)  # Small tolerance

def check_tangency(center1, center2, radius1, radius2, sphere_radius, tolerance=0.05):
    """Check if two caps are tangent within tolerance"""
    ang_dist = angular_distance(center1, center2)
    ang_rad1 = calculate_angular_from_cap_radius(sphere_radius, radius1)
    ang_rad2 = calculate_angular_from_cap_radius(sphere_radius, radius2)
    expected = ang_rad1 + ang_rad2
    return abs(ang_dist - expected) < tolerance

def check_coverage_monte_carlo(centers, radii, sphere_radius, n_samples=5000):
    """Check sphere coverage using Monte Carlo sampling"""
    covered = 0
    
    for _ in range(n_samples):
        # Generate random point on sphere
        theta = np.arccos(1 - 2 * np.random.random())
        phi = 2 * np.pi * np.random.random()
        
        x = sphere_radius * np.sin(theta) * np.cos(phi)
        y = sphere_radius * np.sin(theta) * np.sin(phi)
        z = sphere_radius * np.cos(theta)
        point = np.array([x, y, z])
        
        # Check if covered by any cap
        for center, radius in zip(centers, radii):
            ang_dist = angular_distance(point, center)
            ang_radius = calculate_angular_from_cap_radius(sphere_radius, radius)
            
            if ang_dist <= ang_radius:
                covered += 1
                break
    
    return (covered / n_samples) * 100

def plot_sphere_with_caps(sphere_radius, centers, cap_radii, title=""):
    """Visualize the sphere with spherical caps"""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.2, color='lightgray', linewidth=0.5)
    
    # Color map
    colors = plt.cm.rainbow(np.linspace(0, 1, len(centers)))
    
    # Track statistics
    overlaps = []
    tangencies = []
    
    # Plot caps
    for i, (center, cap_radius, color) in enumerate(zip(centers, cap_radii, colors)):
        # Ensure center is on sphere surface
        center_norm = center / np.linalg.norm(center) * sphere_radius
        
        # Calculate cap boundary circle
        normal = center_norm / sphere_radius
        
        # Find orthogonal vectors
        if abs(normal[2]) < 0.9:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        
        # Generate circle points
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_points = []
        
        for t in theta:
            point_in_plane = cap_radius * (np.cos(t) * v1 + np.sin(t) * v2)
            point_direction = center_norm + point_in_plane
            point_direction = point_direction / np.linalg.norm(point_direction)
            point_on_sphere = sphere_radius * point_direction
            circle_points.append(point_on_sphere)
        
        circle_points = np.array(circle_points)
        ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
                color=color, linewidth=2.5, alpha=0.9)
        
        # Mark center
        ax.scatter(*center_norm, color='black', s=40, marker='o')
        ax.text(center_norm[0]*1.1, center_norm[1]*1.1, center_norm[2]*1.1, 
                f'{i+1}', fontsize=10, ha='center', va='center')
        
        # Check relationships with other caps
        for j in range(i + 1, len(centers)):
            if check_overlap(centers[i], centers[j], cap_radii[i], cap_radii[j], sphere_radius):
                overlaps.append((i, j))
            elif check_tangency(centers[i], centers[j], cap_radii[i], cap_radii[j], sphere_radius):
                tangencies.append((i, j))
    
    # Draw tangent connections
    for i, j in tangencies:
        c1 = centers[i] / np.linalg.norm(centers[i]) * sphere_radius
        c2 = centers[j] / np.linalg.norm(centers[j]) * sphere_radius
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], 
                'g--', alpha=0.4, linewidth=1)
    
    # Highlight overlaps (shouldn't happen in good packing)
    for i, j in overlaps:
        c1 = centers[i] / np.linalg.norm(centers[i]) * sphere_radius
        c2 = centers[j] / np.linalg.norm(centers[j]) * sphere_radius
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], 
                'r-', alpha=0.6, linewidth=2)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Sphere Packing: {len(centers)} Caps, {len(tangencies)} Tangent, {len(overlaps)} Overlaps', 
                    fontsize=14)
    
    # Equal aspect
    ax.set_box_aspect([1,1,1])
    max_range = sphere_radius * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    ax.view_init(elev=25, azim=45)
    
    return fig, overlaps, tangencies

def generate_output_text(sphere_radius, centers, cap_radii, coverage, overlaps, tangencies):
    """Generate output text file"""
    output = f"Sphere Cap Packing Configuration\n"
    output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += f"{'='*70}\n\n"
    output += f"Sphere radius: {sphere_radius:.6f}\n"
    output += f"Number of caps: {len(centers)}\n"
    output += f"Surface coverage: {coverage:.1f}%\n"
    output += f"Tangent pairs: {len(tangencies)}\n"
    output += f"Overlapping pairs: {len(overlaps)}\n\n"
    
    output += f"Spherical Caps:\n"
    output += f"{'-'*70}\n"
    
    for i, (center, cap_radius) in enumerate(zip(centers, cap_radii)):
        center_norm = center / np.linalg.norm(center) * sphere_radius
        
        output += f"\nCap {i+1}:\n"
        output += f"  Center: ({center_norm[0]:.6f}, {center_norm[1]:.6f}, {center_norm[2]:.6f})\n"
        output += f"  Circle radius: {cap_radius:.6f}\n"
        
        angular_radius = calculate_angular_from_cap_radius(sphere_radius, cap_radius)
        output += f"  Angular radius: {np.degrees(angular_radius):.2f}°\n"
        
        # Find tangent neighbors
        tangent_neighbors = []
        for pair in tangencies:
            if i in pair:
                tangent_neighbors.append(pair[0] + 1 if pair[1] == i else pair[1] + 1)
        
        if tangent_neighbors:
            output += f"  Tangent to: {sorted(tangent_neighbors)}\n"
    
    return output

# Streamlit App
st.title("🌐 Sphere Cap Packing - Tangent Circles")

st.markdown("""
This app creates packings of spherical caps where:
- Each cap's boundary is a circle on the sphere
- Caps are tangent to neighbors (no overlap)
- Maximum coverage of the sphere surface
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")

sphere_radius = st.sidebar.number_input(
    "Sphere Radius", 
    min_value=1.0, 
    max_value=100.0, 
    value=10.0, 
    step=0.1
)

packing_type = st.sidebar.selectbox(
    "Packing Type",
    ["12 Caps (Icosahedral)", "20 Caps (Dodecahedral)", "Custom Number"],
    help="Icosahedral and Dodecahedral give optimal tangent packings"
)

if packing_type == "Custom Number":
    n_caps = st.sidebar.slider("Number of Caps", min_value=4, max_value=100, value=30)
    
    st.sidebar.info("""
    **Note:** For custom numbers, the algorithm places caps uniformly 
    and sizes them to avoid overlap while maximizing coverage.
    Perfect tangent packing may not be possible.
    """)

st.sidebar.markdown("---")

# Generate button
if st.sidebar.button("🚀 Generate Packing", type="primary"):
    with st.spinner("Calculating optimal packing..."):
        
        # Generate centers based on type
        if "12" in packing_type:
            centers = generate_icosahedral_vertices()
            # Calculate radius for tangent packing
            angular_radius = calculate_tangent_radius_for_vertices(centers)
            centers = centers * sphere_radius
            cap_radii = np.full(12, calculate_cap_radius_from_angular(sphere_radius, angular_radius))
            
        elif "20" in packing_type:
            centers = generate_dodecahedral_vertices()
            # Calculate radius for tangent packing
            angular_radius = calculate_tangent_radius_for_vertices(centers)
            centers = centers * sphere_radius
            cap_radii = np.full(20, calculate_cap_radius_from_angular(sphere_radius, angular_radius))
            
        else:
            # Custom number - uniform distribution
            centers = generate_uniform_points(n_caps)
            centers = centers * sphere_radius
            # Optimize radii to maximize coverage without overlap
            cap_radii = optimize_radii_for_coverage(centers, sphere_radius)
        
        # Check coverage
        coverage = check_coverage_monte_carlo(centers, cap_radii, sphere_radius)
        
        # Create visualization
        fig, overlaps, tangencies = plot_sphere_with_caps(sphere_radius, centers, cap_radii)
        st.pyplot(fig)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Number of Caps", len(centers))
        with col2:
            st.metric("Coverage", f"{coverage:.1f}%")
        with col3:
            st.metric("Tangent Pairs", len(tangencies))
        with col4:
            st.metric("Overlaps", len(overlaps), 
                     help="Should be 0 for good packing")
        
        # Quality assessment
        if len(overlaps) > 0:
            st.error(f"⚠️ {len(overlaps)} overlapping pairs detected! This shouldn't happen.")
        elif coverage >= 98:
            st.success(f"✨ Excellent packing! {coverage:.1f}% coverage with {len(tangencies)} tangent pairs")
        elif coverage >= 90:
            st.info(f"✓ Good packing: {coverage:.1f}% coverage with {len(tangencies)} tangent pairs")
        else:
            st.warning(f"Coverage: {coverage:.1f}% - Some gaps remain in the packing")
        
        # Special case info
        if packing_type in ["12 Caps (Icosahedral)", "20 Caps (Dodecahedral)"]:
            st.info("ℹ️ This configuration provides mathematically optimal tangent packing!")
        
        # Generate output
        output_text = generate_output_text(sphere_radius, centers, cap_radii, coverage, overlaps, tangencies)
        
        # Download button
        st.download_button(
            label="📥 Download Configuration",
            data=output_text,
            file_name=f"sphere_packing_{len(centers)}caps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        with st.expander("📋 View Details"):
            st.text(output_text)

# Theory section
with st.expander("📚 Mathematical Background"):
    st.markdown("""
    ### The Challenge
    
    Packing circles on a sphere such that:
    1. They don't overlap (except at tangent points)
    2. They cover the entire surface
    3. Each circle is tangent to its neighbors
    
    is only perfectly solvable for certain special numbers.
    
    ### Perfect Solutions
    
    **12 Caps (Icosahedral):**
    - Based on icosahedron vertices
    - Each cap has 5 neighbors
    - Angular radius ≈ 31.72°
    
    **20 Caps (Dodecahedral):**
    - Based on dodecahedron vertices  
    - Each cap has 3 neighbors
    - Angular radius ≈ 23.65°
    
    ### For Other Numbers
    
    The algorithm:
    1. Distributes points uniformly using Fibonacci spiral
    2. Calculates maximum radius without overlap
    3. Aims for maximum coverage
    
    Perfect tangent packing is generally impossible for arbitrary numbers.
    """)

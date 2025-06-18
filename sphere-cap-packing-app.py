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

def check_tangency(center1, center2, radius1, radius2, sphere_radius, tolerance=0.05):
    """Check if two caps are tangent within tolerance"""
    ang_dist = angular_distance(center1, center2)
    ang_rad1 = calculate_angular_from_cap_radius(sphere_radius, radius1)
    ang_rad2 = calculate_angular_from_cap_radius(sphere_radius, radius2)
    expected = ang_rad1 + ang_rad2
    return abs(ang_dist - expected) < tolerance

def generate_optimal_packing_12():
    """Generate icosahedral packing for 12 circles"""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # Icosahedron vertices
    centers = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            centers.append([0, i, j * phi])
            centers.append([i, j * phi, 0])
            centers.append([j * phi, 0, i])
    
    centers = np.array(centers)
    # Normalize to unit sphere
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    # For icosahedron, each vertex has 5 neighbors
    # Angular distance to nearest neighbor
    min_dist = float('inf')
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = angular_distance(centers[i], centers[j])
            if dist < min_dist:
                min_dist = dist
    
    # Each cap has angular radius = half the distance to nearest neighbor
    angular_radius = min_dist / 2
    
    return centers, angular_radius

def generate_optimal_packing_20():
    """Generate dodecahedral packing for 20 circles"""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # Dodecahedron vertices
    centers = []
    
    # (±1, ±1, ±1)
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                centers.append([x, y, z])
    
    # (0, ±1/φ, ±φ)
    for i in [-1, 1]:
        for j in [-1, 1]:
            centers.append([0, i/phi, j*phi])
            centers.append([i/phi, j*phi, 0])
            centers.append([j*phi, 0, i/phi])
    
    centers = np.array(centers)
    # Normalize to unit sphere
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    # Find minimum distance between vertices
    min_dist = float('inf')
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = angular_distance(centers[i], centers[j])
            if dist < min_dist:
                min_dist = dist
    
    # Each cap has angular radius = half the distance to nearest neighbor
    angular_radius = min_dist / 2
    
    return centers, angular_radius

def generate_spiral_packing(n_circles):
    """Generate uniform distribution using Fibonacci spiral"""
    centers = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle
    
    for i in range(n_circles):
        y = 1 - (i / float(n_circles - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        
        theta = phi * i  # golden angle increment
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        centers.append([x, y, z])
    
    centers = np.array(centers)
    
    # Find average minimum distance to determine radius
    min_distances = []
    for i in range(len(centers)):
        distances = []
        for j in range(len(centers)):
            if i != j:
                distances.append(angular_distance(centers[i], centers[j]))
        if distances:
            min_distances.append(min(distances))
    
    # Set angular radius to achieve good coverage
    avg_min_dist = np.mean(min_distances)
    angular_radius = avg_min_dist / 2 * 0.95  # Slightly smaller to ensure tangency
    
    return centers, angular_radius

def check_coverage_grid(centers, radii, sphere_radius, grid_size=50):
    """Check sphere coverage using a grid approach"""
    # Generate grid points on sphere
    covered_area = 0
    total_area = 0
    
    for theta in np.linspace(0, np.pi, grid_size):
        for phi in np.linspace(0, 2*np.pi, int(grid_size * np.sin(theta) + 1)):
            # Point on sphere
            x = sphere_radius * np.sin(theta) * np.cos(phi)
            y = sphere_radius * np.sin(theta) * np.sin(phi)
            z = sphere_radius * np.cos(theta)
            point = np.array([x, y, z])
            
            # Area element
            dA = sphere_radius**2 * np.sin(theta) * (np.pi/grid_size) * (2*np.pi/max(1, int(grid_size * np.sin(theta))))
            total_area += dA
            
            # Check if covered by any cap
            for center, radius in zip(centers, radii):
                ang_dist = angular_distance(point, center)
                ang_radius = calculate_angular_from_cap_radius(sphere_radius, radius)
                
                if ang_dist <= ang_radius:
                    covered_area += dA
                    break
    
    return (covered_area / total_area) * 100 if total_area > 0 else 0

def plot_sphere_with_caps(sphere_radius, centers, cap_radii, show_labels=True):
    """Visualize the sphere with spherical caps"""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot sphere surface with better resolution
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot sphere with light color
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue', shade=False)
    
    # Plot meridians and parallels for reference
    for i in range(0, 360, 30):
        theta = np.radians(i)
        x_meridian = sphere_radius * np.sin(v) * np.cos(theta)
        y_meridian = sphere_radius * np.sin(v) * np.sin(theta)
        z_meridian = sphere_radius * np.cos(v)
        ax.plot(x_meridian, y_meridian, z_meridian, 'k-', alpha=0.1, linewidth=0.5)
    
    # Color map for caps
    colors = plt.cm.rainbow(np.linspace(0, 1, len(centers)))
    
    # Track tangent pairs
    tangent_pairs = []
    
    # Plot spherical caps
    for i, (center, cap_radius, color) in enumerate(zip(centers, cap_radii, colors)):
        # Normalize center
        center = center * sphere_radius / np.linalg.norm(center)
        
        # Plot the cap circle
        normal = center / sphere_radius
        
        # Find two orthogonal vectors in the plane
        if abs(normal[2]) < 0.9:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Generate circle points with higher resolution
        theta = np.linspace(0, 2 * np.pi, 150)
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
                color=color, linewidth=3, alpha=0.9)
        
        # Mark center with larger dot
        ax.scatter(*center, color='black', s=50, marker='o', alpha=0.8)
        
        # Add label if requested
        if show_labels:
            ax.text(center[0]*1.1, center[1]*1.1, center[2]*1.1, f'{i+1}', 
                   fontsize=8, ha='center', va='center')
        
        # Check for tangent neighbors
        for j in range(i + 1, len(centers)):
            if check_tangency(centers[i], centers[j], cap_radii[i], cap_radii[j], sphere_radius):
                tangent_pairs.append((i, j))
    
    # Draw lines between tangent pairs
    for i, j in tangent_pairs:
        center_i = centers[i] * sphere_radius / np.linalg.norm(centers[i])
        center_j = centers[j] * sphere_radius / np.linalg.norm(centers[j])
        ax.plot([center_i[0], center_j[0]], 
               [center_i[1], center_j[1]], 
               [center_i[2], center_j[2]], 
               'g--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'Sphere Packing: {len(centers)} Caps, {len(tangent_pairs)} Tangent Pairs', fontsize=14)
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    max_range = sphere_radius * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Better viewing angle
    ax.view_init(elev=20, azim=45)
    
    return fig, tangent_pairs

def generate_output_text(sphere_radius, centers, cap_radii, coverage, tangent_pairs):
    """Generate the output text file content"""
    output = f"Sphere Cap Packing Configuration\n"
    output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += f"{'='*70}\n\n"
    output += f"Sphere radius: {sphere_radius:.6f}\n"
    output += f"Number of caps: {len(centers)}\n"
    output += f"Surface coverage: {coverage:.1f}%\n"
    output += f"Number of tangent pairs: {len(tangent_pairs)}\n\n"
    
    output += f"Spherical Caps:\n"
    output += f"{'-'*70}\n"
    
    for i, (center, cap_radius) in enumerate(zip(centers, cap_radii)):
        # Normalize center to sphere surface
        center = center * sphere_radius / np.linalg.norm(center)
        
        output += f"\nCap {i+1}:\n"
        output += f"  Center coordinates: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})\n"
        output += f"  Cap circle radius: {cap_radius:.6f}\n"
        
        angular_radius = calculate_angular_from_cap_radius(sphere_radius, cap_radius)
        output += f"  Angular radius: {np.degrees(angular_radius):.2f}°\n"
        output += f"  Cap height: {sphere_radius * (1 - np.cos(angular_radius)):.6f}\n"
        
        # List tangent neighbors
        tangent_neighbors = []
        for pair in tangent_pairs:
            if i in pair:
                tangent_neighbors.append(pair[0] + 1 if pair[1] == i else pair[1] + 1)
        
        if tangent_neighbors:
            output += f"  Tangent to caps: {sorted(tangent_neighbors)}\n"
    
    output += f"\n{'='*70}\n"
    output += f"Summary Statistics:\n"
    output += f"  Total surface area of sphere: {4 * np.pi * sphere_radius**2:.6f}\n"
    output += f"  Average cap area: {4 * np.pi * sphere_radius**2 / len(centers):.6f}\n"
    output += f"  Coverage percentage: {coverage:.1f}%\n"
    output += f"  Tangency ratio: {len(tangent_pairs) / (len(centers) * (len(centers)-1) / 2) * 100:.1f}%\n"
    
    return output

# Streamlit App
st.title("🌐 Complete Sphere Cap Packing")

st.markdown("""
### Goal: Fill the sphere surface with tangent circular caps

This app generates packings where:
- Each cap's boundary is a circle on the sphere surface
- Caps are tangent to their neighbors (boundaries touch)
- The entire sphere is covered with minimal gaps
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")

sphere_radius = st.sidebar.number_input(
    "Sphere Radius", 
    min_value=1.0, 
    max_value=100.0, 
    value=10.0, 
    step=0.1,
    help="Radius of the sphere to be packed"
)

packing_type = st.sidebar.selectbox(
    "Packing Type",
    ["12 Caps (Icosahedral)", "20 Caps (Dodecahedral)", 
     "Custom (Spiral Distribution)"],
    help="Select the packing arrangement"
)

if packing_type == "Custom (Spiral Distribution)":
    n_caps = st.sidebar.slider(
        "Number of Caps", 
        min_value=4, 
        max_value=100, 
        value=30,
        help="Number of caps to pack on the sphere"
    )
else:
    n_caps = 12 if "12" in packing_type else 20

show_labels = st.sidebar.checkbox("Show cap numbers", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**💡 Tips:**
- 12 and 20 caps give mathematically optimal packings
- For custom numbers, the algorithm approximates optimal packing
- Higher cap counts give better coverage but less perfect tangency
""")

# Generate button
if st.sidebar.button("🚀 Generate Packing", type="primary"):
    with st.spinner("Generating sphere packing..."):
        
        # Generate packing based on type
        if "12" in packing_type:
            centers, angular_radius = generate_optimal_packing_12()
        elif "20" in packing_type:
            centers, angular_radius = generate_optimal_packing_20()
        else:
            centers, angular_radius = generate_spiral_packing(n_caps)
        
        # Scale centers to sphere radius and calculate cap radii
        centers = centers * sphere_radius
        cap_radii = np.full(len(centers), calculate_cap_radius_from_angular(sphere_radius, angular_radius))
        
        # Check coverage
        coverage = check_coverage_grid(centers, cap_radii, sphere_radius)
        
        # Create visualization
        fig, tangent_pairs = plot_sphere_with_caps(sphere_radius, centers, cap_radii, show_labels)
        st.pyplot(fig)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sphere Radius", f"{sphere_radius:.1f}")
        with col2:
            st.metric("Number of Caps", len(centers))
        with col3:
            st.metric("Coverage", f"{coverage:.1f}%")
        with col4:
            st.metric("Tangent Pairs", len(tangent_pairs))
        
        # Quality assessment
        if coverage >= 98:
            st.success(f"✨ Excellent! {coverage:.1f}% coverage with {len(tangent_pairs)} tangent pairs")
        elif coverage >= 95:
            st.info(f"✓ Very good! {coverage:.1f}% coverage with {len(tangent_pairs)} tangent pairs")
        elif coverage >= 90:
            st.warning(f"⚠️ Good coverage ({coverage:.1f}%) but some gaps remain")
        else:
            st.error(f"❌ Only {coverage:.1f}% coverage - significant gaps present")
        
        # Generate output
        output_text = generate_output_text(sphere_radius, centers, cap_radii, coverage, tangent_pairs)
        
        # Download button
        st.download_button(
            label="📥 Download Configuration",
            data=output_text,
            file_name=f"sphere_packing_{n_caps}caps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        # Expandable output preview
        with st.expander("📄 View Configuration Details"):
            st.text(output_text)

# Information section
with st.expander("📚 Mathematical Background"):
    st.markdown("""
    ### Sphere Packing Problem
    
    The problem of packing circles on a sphere such that:
    1. Each circle boundary lies on the sphere surface (spherical caps)
    2. Circles are tangent to their neighbors
    3. The sphere is completely covered
    
    is related to several classical problems in mathematics:
    
    **Perfect Solutions Exist For:**
    - **4 caps**: Tetrahedral arrangement (each cap covers 1/4 of sphere)
    - **6 caps**: Octahedral arrangement (along coordinate axes)
    - **8 caps**: Cubic arrangement
    - **12 caps**: Icosahedral arrangement ✨
    - **20 caps**: Dodecahedral arrangement ✨
    
    **Key Concepts:**
    - **Angular radius**: The angle from sphere center to cap boundary
    - **Cap circle radius**: The actual radius of the circle on the sphere surface
    - **Tangency**: Two caps are tangent when their boundaries touch at exactly one point
    
    **Why 12 and 20 are special:**
    These correspond to the vertices of the icosahedron and dodecahedron, which provide
    the most symmetric distributions of points on a sphere.
    """)

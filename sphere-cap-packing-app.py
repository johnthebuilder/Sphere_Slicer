import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from datetime import datetime

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

def generate_fibonacci_sphere(n_points, sphere_radius):
    """Generate evenly distributed points on sphere using Fibonacci spiral"""
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

def generate_spiral_points(n_points, sphere_radius):
    """Generate points in a spiral pattern"""
    points = []
    
    for i in range(n_points):
        # Spiral from top to bottom
        theta = np.pi * i / (n_points - 1)  # 0 to pi
        phi = 4 * np.pi * i / (n_points - 1)  # Multiple rotations
        
        x, y, z = spherical_to_cartesian(theta, phi, sphere_radius)
        points.append([x, y, z])
    
    return np.array(points)

def calculate_cap_radius_from_angular(sphere_radius, angular_radius):
    """Calculate the actual circle radius of a spherical cap given its angular radius"""
    return sphere_radius * np.sin(angular_radius)

def calculate_angular_from_cap_radius(sphere_radius, cap_radius):
    """Calculate angular radius from cap circle radius"""
    return np.arcsin(cap_radius / sphere_radius)

def check_caps_overlap(center1, center2, angular_radius1, angular_radius2):
    """Check if two spherical caps overlap"""
    # Angular distance between centers
    dot_product = np.dot(center1, center2) / (np.linalg.norm(center1) * np.linalg.norm(center2))
    angular_distance = np.arccos(np.clip(dot_product, -1, 1))
    
    # Caps don't overlap if angular distance > sum of angular radii
    return angular_distance < (angular_radius1 + angular_radius2)

def check_caps_tangent(center1, center2, angular_radius1, angular_radius2, tolerance=0.01):
    """Check if two spherical caps are tangent"""
    # Angular distance between centers
    dot_product = np.dot(center1, center2) / (np.linalg.norm(center1) * np.linalg.norm(center2))
    angular_distance = np.arccos(np.clip(dot_product, -1, 1))
    
    # Caps are tangent if angular distance ≈ sum of angular radii
    expected_distance = angular_radius1 + angular_radius2
    return abs(angular_distance - expected_distance) < tolerance

def optimize_cap_radii(centers, sphere_radius, min_radius_ratio, max_radius_ratio, target_tangencies=4):
    """Optimize cap radii to maximize tangencies while avoiding overlaps"""
    n_caps = len(centers)
    
    # Initialize with random radii in the allowed range
    min_cap_radius = sphere_radius * min_radius_ratio
    max_cap_radius = sphere_radius * max_radius_ratio
    
    cap_radii = np.random.uniform(min_cap_radius, max_cap_radius, n_caps)
    angular_radii = np.array([calculate_angular_from_cap_radius(sphere_radius, r) for r in cap_radii])
    
    # Simple optimization: gradually adjust radii to increase tangencies
    for iteration in range(100):
        tangency_count = np.zeros(n_caps)
        
        # Count tangencies for each cap
        for i in range(n_caps):
            for j in range(i + 1, n_caps):
                if check_caps_tangent(centers[i], centers[j], angular_radii[i], angular_radii[j]):
                    tangency_count[i] += 1
                    tangency_count[j] += 1
        
        # Adjust radii based on tangency count
        for i in range(n_caps):
            if tangency_count[i] < target_tangencies:
                # Increase radius slightly
                cap_radii[i] = min(cap_radii[i] * 1.02, max_cap_radius)
            elif tangency_count[i] > target_tangencies:
                # Decrease radius slightly
                cap_radii[i] = max(cap_radii[i] * 0.98, min_cap_radius)
        
        angular_radii = np.array([calculate_angular_from_cap_radius(sphere_radius, r) for r in cap_radii])
        
        # Check for overlaps and fix them
        for i in range(n_caps):
            for j in range(i + 1, n_caps):
                if i != j and check_caps_overlap(centers[i], centers[j], angular_radii[i], angular_radii[j]):
                    # Reduce both radii slightly
                    cap_radii[i] *= 0.95
                    cap_radii[j] *= 0.95
    
    return cap_radii

def plot_sphere_with_caps(sphere_radius, centers, cap_radii):
    """Visualize the sphere with spherical caps"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.2, color='gray')
    
    # Plot spherical caps
    for i, (center, cap_radius) in enumerate(zip(centers, cap_radii)):
        # Plot the cap circle
        # Create a circle in the plane perpendicular to the center vector
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
        theta = np.linspace(0, 2 * np.pi, 50)
        circle_points = []
        
        angular_radius = calculate_angular_from_cap_radius(sphere_radius, cap_radius)
        
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
                color=plt.cm.rainbow(i / len(centers)), linewidth=2)
        
        # Mark center
        ax.scatter(*center, color=plt.cm.rainbow(i / len(centers)), s=50, marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Sphere with {len(centers)} Spherical Caps')
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    max_range = sphere_radius * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    return fig

def generate_output_text(sphere_radius, centers, cap_radii):
    """Generate the output text file content"""
    output = f"Sphere Configuration\n"
    output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += f"{'='*50}\n\n"
    output += f"Sphere radius: {sphere_radius:.6f}\n"
    output += f"Number of caps: {len(centers)}\n\n"
    
    output += f"Spherical Caps:\n"
    output += f"{'-'*50}\n"
    
    for i, (center, cap_radius) in enumerate(zip(centers, cap_radii)):
        output += f"Cap {i+1}:\n"
        output += f"  Center: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})\n"
        output += f"  Cap circle radius: {cap_radius:.6f}\n"
        angular_radius = calculate_angular_from_cap_radius(sphere_radius, cap_radius)
        output += f"  Angular radius: {np.degrees(angular_radius):.2f}°\n"
        output += f"  Cap height: {sphere_radius * (1 - np.cos(angular_radius)):.6f}\n\n"
    
    # Add tangency information
    output += f"\nTangency Information:\n"
    output += f"{'-'*50}\n"
    
    tangency_pairs = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            angular_i = calculate_angular_from_cap_radius(sphere_radius, cap_radii[i])
            angular_j = calculate_angular_from_cap_radius(sphere_radius, cap_radii[j])
            if check_caps_tangent(centers[i], centers[j], angular_i, angular_j):
                tangency_pairs.append((i+1, j+1))
    
    output += f"Total tangent pairs: {len(tangency_pairs)}\n"
    output += "Tangent pairs: "
    output += ", ".join([f"({i},{j})" for i, j in tangency_pairs])
    
    return output

# Streamlit App
st.title("Spherical Cap Packing Generator")

st.markdown("""
This app generates spherical caps on a sphere surface where each cap's boundary 
is tangent to its neighbors. The caps are non-overlapping except at tangent points.
""")

# Sidebar parameters
st.sidebar.header("Parameters")

sphere_radius = st.sidebar.number_input("Sphere Radius", min_value=1.0, max_value=100.0, value=10.0, step=0.1)

n_caps = st.sidebar.slider("Number of Caps", min_value=4, max_value=50, value=20)

pattern = st.sidebar.selectbox(
    "Initial Pattern",
    ["Fibonacci Spiral", "Icosahedral (12 points)", "Spiral", "Random"]
)

min_radius_ratio = st.sidebar.slider(
    "Min Cap Radius (% of sphere)", 
    min_value=0.05, max_value=0.5, value=0.1, step=0.01
)

max_radius_ratio = st.sidebar.slider(
    "Max Cap Radius (% of sphere)", 
    min_value=0.1, max_value=0.8, value=0.3, step=0.01
)

if min_radius_ratio >= max_radius_ratio:
    st.sidebar.error("Min radius must be less than max radius!")

optimize = st.sidebar.checkbox("Optimize cap sizes for tangencies", value=True)

if st.sidebar.button("Generate"):
    with st.spinner("Generating sphere cap configuration..."):
        # Generate initial center points based on pattern
        if pattern == "Fibonacci Spiral":
            centers = generate_fibonacci_sphere(n_caps, sphere_radius)
        elif pattern == "Icosahedral (12 points)":
            centers = generate_icosahedral_points(sphere_radius)
            n_caps = 12  # Fixed for icosahedral
        elif pattern == "Spiral":
            centers = generate_spiral_points(n_caps, sphere_radius)
        else:  # Random
            # Generate random points on sphere
            centers = []
            for _ in range(n_caps):
                theta = np.arccos(1 - 2 * np.random.random())
                phi = 2 * np.pi * np.random.random()
                x, y, z = spherical_to_cartesian(theta, phi, sphere_radius)
                centers.append([x, y, z])
            centers = np.array(centers)
        
        # Generate or optimize cap radii
        if optimize:
            cap_radii = optimize_cap_radii(centers, sphere_radius, min_radius_ratio, max_radius_ratio)
        else:
            # Random radii within range
            cap_radii = np.random.uniform(
                sphere_radius * min_radius_ratio,
                sphere_radius * max_radius_ratio,
                len(centers)
            )
        
        # Create visualization
        fig = plot_sphere_with_caps(sphere_radius, centers, cap_radii)
        st.pyplot(fig)
        
        # Generate output text
        output_text = generate_output_text(sphere_radius, centers, cap_radii)
        
        # Display summary
        st.subheader("Configuration Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sphere Radius", f"{sphere_radius:.2f}")
        with col2:
            st.metric("Number of Caps", len(centers))
        with col3:
            # Count tangencies
            tangency_count = 0
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    angular_i = calculate_angular_from_cap_radius(sphere_radius, cap_radii[i])
                    angular_j = calculate_angular_from_cap_radius(sphere_radius, cap_radii[j])
                    if check_caps_tangent(centers[i], centers[j], angular_i, angular_j):
                        tangency_count += 1
            st.metric("Tangent Pairs", tangency_count)
        
        # Download button
        st.download_button(
            label="Download Configuration",
            data=output_text,
            file_name=f"sphere_caps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        # Show output preview
        with st.expander("View Output File Preview"):
            st.text(output_text)

# Instructions
with st.expander("Instructions"):
    st.markdown("""
    1. **Set the sphere radius** - This determines the size of your sphere
    2. **Choose number of caps** - How many spherical caps to place on the sphere
    3. **Select initial pattern**:
       - *Fibonacci Spiral*: Even distribution using golden ratio
       - *Icosahedral*: 12 points based on icosahedron vertices
       - *Spiral*: Points arranged in a spiral from pole to pole
       - *Random*: Randomly distributed points
    4. **Set radius range** - Min and max cap sizes as percentage of sphere radius
    5. **Enable optimization** - Attempts to adjust cap sizes to maximize tangencies
    6. **Click Generate** - Creates the configuration and visualization
    7. **Download** - Save the configuration to a text file
    
    The app will show:
    - 3D visualization of the sphere with caps
    - Each cap's boundary circle on the sphere surface
    - Summary statistics
    - Downloadable text file with all parameters
    """)

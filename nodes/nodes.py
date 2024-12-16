import torch
import numpy as np
from PIL import Image, ImageDraw

class MaskContourProcessor:
    """A class that adds decorative flame-like contour effects to binary masks.
    
    This processor detects mask edges and generates flame-ray effects along the contour,
    creating visually appealing decorative patterns that radiate from the mask's edges.
    """
    
    # Effect generation constants
    AMPLITUDE_INITIAL_MOD = 0.2
    AMPLITUDE_DECAY_MOD = 0.8
    MIN_LINE_WIDTH = 1.0
    DEFAULT_STROKE_COLOR = 1.0  # White (1.0) for mask
    DEFAULT_ELEMENT_COLOR = 1.0  # White (1.0) for mask
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "line_length": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 3.0,
                    "step": 0.01
                }),
                "line_count": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 40,
                    "step": 1
                }),
                "line_width": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.001
                })
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process_mask"
    CATEGORY = "mask"

    def __init__(self):
        self.stroke_color = self.DEFAULT_STROKE_COLOR
        self.element_color = self.DEFAULT_ELEMENT_COLOR

    def set_colors(self, stroke_color, element_color):
        """Configure the colors used for rendering effects.

        Args:
            stroke_color (str): Color to use for the main flame strokes
            element_color (str): Color to use for decorative elements
        """
        self.stroke_color = stroke_color
        self.element_color = element_color

    def calculate_mask_centroid(self, mask):
        """Calculate the centroid of the mask."""
        # Get coordinates of white pixels (mask > 0.5)
        coords = np.nonzero(mask > 0.5)
        
        if len(coords[0]) == 0:  # If mask is empty
            height, width = mask.shape
            return width // 2, height // 2
        
        # Calculate centroid from white pixel coordinates
        y_coords, x_coords = coords
        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)
        
        return center_x, center_y

    def detect_edge_points(self, mask_array, center):
        """Detect and sort edge points of a mask in clockwise order.

        Args:
            mask_array (ndarray): Binary mask array
            center (tuple): (x, y) coordinates of the mask's centroid

        Returns:
            list: Edge points as [(x, y), ...] sorted clockwise around the center
        """
        edges = set()
        height, width = mask_array.shape
        
        # Scan horizontally (left to right) - looking for transitions between 0 and 1
        for y in range(height):
            last_pixel = 0
            for x in range(width):
                current_pixel = 1 if mask_array[y, x] > 0.5 else 0
                if last_pixel != current_pixel:
                    edges.add((x, y))
                last_pixel = current_pixel
                
        # Scan vertically (top to bottom)
        for x in range(width):
            last_pixel = 0
            for y in range(height):
                current_pixel = 1 if mask_array[y, x] > 0.5 else 0
                if last_pixel != current_pixel:
                    edges.add((x, y))
                last_pixel = current_pixel
        
        # Convert to list and sort points clockwise around center
        edge_points = list(edges)
        center_x, center_y = center
        
        # Sort points clockwise using arctan2
        edge_points.sort(key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))
        
        return edge_points

    def redistribute_points(self, points, target_count):
        """Redistribute points evenly along a contour path.

        Args:
            points (list): List of (x, y) points defining the contour
            target_count (int): Desired number of evenly spaced points

        Returns:
            list: New list of (x, y) points evenly distributed along the contour
        """
        # Calculate total perimeter length
        total_length = 0
        segments = []
        
        for i in range(len(points)):
            next_point = points[(i + 1) % len(points)]
            length = np.sqrt(
                (next_point[0] - points[i][0])**2 + 
                (next_point[1] - points[i][1])**2
            )
            total_length += length
            segments.append({
                'start': points[i],
                'end': next_point,
                'length': length
            })

        # Redistribute points evenly
        spacing = total_length / target_count
        new_points = []
        current_dist = 0
        current_segment = 0
        
        for i in range(target_count):
            # Skip to next segment if we've exceeded current segment length
            while current_dist >= segments[current_segment]['length']:
                current_dist -= segments[current_segment]['length']
                current_segment = (current_segment + 1) % len(segments)
                
            # Calculate interpolation factor for current position
            seg = segments[current_segment]
            t = current_dist / seg['length']
            
            # Interpolate position
            new_points.append((
                seg['start'][0] + (seg['end'][0] - seg['start'][0]) * t,
                seg['start'][1] + (seg['end'][1] - seg['start'][1]) * t
            ))
            
            current_dist += spacing

        return new_points

    def calculate_base_line_width(self, edge_points, line_width):
        """Calculate the base width for effect lines based on contour size.

        Args:
            edge_points (list): List of (x, y) edge points
            line_width (float): Width parameter for scaling

        Returns:
            float: Calculated base line width
        """
        if not edge_points:
            return 10  # Default fallback value
        
        # Calculate circumference by summing distances between consecutive points
        circumference = 0
        for i in range(len(edge_points)):
            current = edge_points[i]
            next_point = edge_points[(i + 1) % len(edge_points)]
            
            distance = np.sqrt(
                (next_point[0] - current[0])**2 + 
                (next_point[1] - current[1])**2
            )
            circumference += distance
        
        # Return max of 1 or circumference * line_width
        return max(1.0, circumference * line_width)

    def generate_flame_ray_effect(self, point, next_point, line_length, center, base_line_width):
        """Generate data for a flame-like ray effect emanating from an edge point.

        Args:
            point (tuple): (x, y) starting point on the contour
            next_point (tuple): (x, y) next point on the contour
            line_length (float): Length of the flame effect
            center (tuple): (x, y) mask centroid
            base_line_width (float): Base width for effect lines

        Returns:
            dict: Effect data containing path segments and decorative elements
        """
        # Calculate direction vectors for current point
        dx1 = point[0] - center[0]
        dy1 = point[1] - center[1]
        length1 = np.sqrt(dx1 * dx1 + dy1 * dy1)
        normalized_dx1 = dx1 / length1
        normalized_dy1 = dy1 / length1

        # Calculate direction vectors for next point
        dx2 = next_point[0] - center[0]
        dy2 = next_point[1] - center[1]
        length2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
        normalized_dx2 = dx2 / length2
        normalized_dy2 = dy2 / length2

        # Calculate midpoint between current and next point
        mid_point = {
            'x': (point[0] + next_point[0]) / 2,
            'y': (point[1] + next_point[1]) / 2
        }

        num_waves = max(5, line_length / 5)
        initial_amplitude = line_length * self.AMPLITUDE_INITIAL_MOD
        min_line_width = self.MIN_LINE_WIDTH

        segments = []
        decorative_elements = []

        # Generate path segments for main flame rays
        for i in range(int(num_waves)):
            t = i / num_waves
            next_t = (i + 1) / num_waves
            
            # Calculate current wave points
            wave_x = point[0] + normalized_dx1 * line_length * t
            wave_y = point[1] + normalized_dy1 * line_length * t
            next_wave_x = point[0] + normalized_dx1 * line_length * next_t
            next_wave_y = point[1] + normalized_dy1 * line_length * next_t

            # Calculate wave amplitudes and offsets
            current_amplitude = initial_amplitude * (1 - t * self.AMPLITUDE_DECAY_MOD)
            offset_x = -normalized_dy1 * np.sin(t * np.pi * 2) * current_amplitude
            offset_y = normalized_dx1 * np.sin(t * np.pi * 2) * current_amplitude
            next_amplitude = initial_amplitude * (1 - next_t * self.AMPLITUDE_DECAY_MOD)
            next_offset_x = -normalized_dy1 * np.sin(next_t * np.pi * 2) * next_amplitude
            next_offset_y = normalized_dx1 * np.sin(next_t * np.pi * 2) * next_amplitude

            # Interpolate between direction vectors for virtual flame
            interpolation_factor = 0.5
            virtual_dx = normalized_dx1 * (1 - interpolation_factor) + normalized_dx2 * interpolation_factor
            virtual_dy = normalized_dy1 * (1 - interpolation_factor) + normalized_dy2 * interpolation_factor
            
            # Normalize interpolated direction
            virtual_length = np.sqrt(virtual_dx * virtual_dx + virtual_dy * virtual_dy)
            normalized_virtual_dx = virtual_dx / virtual_length
            normalized_virtual_dy = virtual_dy / virtual_length

            # Calculate virtual flame point
            virtual_flame_x = mid_point['x'] + normalized_virtual_dx * line_length * t
            virtual_flame_y = mid_point['y'] + normalized_virtual_dy * line_length * t

            # Add sinusoidal pattern to virtual flame
            virtual_amplitude = current_amplitude
            virtual_offset_x = -normalized_virtual_dy * np.sin(t * np.pi * 2) * virtual_amplitude
            virtual_offset_y = normalized_virtual_dx * np.sin(t * np.pi * 2) * virtual_amplitude

            segments.append({
                'startPoint': {
                    'x': wave_x + offset_x,
                    'y': wave_y + offset_y
                },
                'endPoint': {
                    'x': next_wave_x + next_offset_x,
                    'y': next_wave_y + next_offset_y
                },
                'properties': {
                    'width': base_line_width - (base_line_width - min_line_width) * t,
                    'strokeColor': self.stroke_color
                }
            })

            # Add decorative elements along virtual flame path
            start_step = 2
            if i >= start_step and i % 3 == 0:
                progress = (i - start_step) / (num_waves - start_step)
                min_radius = min_line_width / 2
                max_radius = base_line_width * 0.6
                current_radius = min_radius + (max_radius - min_radius) * progress
                
                decorative_elements.append({
                    'position': {
                        'x': virtual_flame_x + virtual_offset_x,
                        'y': virtual_flame_y + virtual_offset_y
                    },
                    'type': 'circle',
                    'properties': {
                        'radius': current_radius,
                        'color': self.element_color
                    }
                })

        return {
            'origin': {'x': point[0], 'y': point[1]},
            'path': {
                'direction': {'dx': normalized_dx1, 'dy': normalized_dy1},
                'length': line_length,
                'segments': segments
            },
            'decorativeElements': decorative_elements
        }

    def render_effect_to_mask(self, mask_array, effect_data):
        """Render a flame effect onto a mask using Pillow for drawing.

        Args:
            mask_array (ndarray): Target mask array
            effect_data (dict): Effect specification data

        Returns:
            ndarray: Updated mask with rendered effect
        """
        height, width = mask_array.shape
        canvas = Image.new('F', (width, height), 0.0)
        draw = ImageDraw.Draw(canvas)

        # Render path segments using Bezier curves
        for segment in effect_data['path']['segments']:
            start = segment['startPoint']
            end = segment['endPoint']
            control = {
                'x': (start['x'] + end['x']) / 2,
                'y': (start['y'] + end['y']) / 2
            }
            line_width = segment['properties']['width']
            
            print(f"Rendering Bezier curve from {start} to {end} with control {control} and width {line_width}")
            
            # Draw Bezier curve using Pillow
            draw.line(
                [(start['x'], start['y']), (control['x'], control['y']), (end['x'], end['y'])],
                fill=self.stroke_color,
                width=int(line_width),
                joint='curve'
            )
        
        # Render decorative elements
        if 'decorativeElements' in effect_data:
            for element in effect_data['decorativeElements']:
                if element['type'] == 'circle':
                    pos = element['position']
                    radius = element['properties']['radius']
                    
                    print(f"Rendering circle at {pos} with radius {radius}")
                    
                    # Draw circle using Pillow
                    bbox = [
                        (pos['x'] - radius, pos['y'] - radius),
                        (pos['x'] + radius, pos['y'] + radius)
                    ]
                    draw.ellipse(bbox, outline=self.element_color, width=1)
        
        # Convert the canvas to a numpy array and combine with the mask
        canvas_np = np.array(canvas)
        combined_mask = np.clip(mask_array + canvas_np, 0, 1)
        return combined_mask

    def process_mask(self, mask, line_length, line_count, line_width):
        """Process a mask by adding flame-like contour effects.

        Args:
            mask (tensor): Input mask tensor
            line_length (float): Relative length of flame effects
            line_count (int): Number of flame effects to generate
            line_width (float): Width of effect lines

        Returns:
            tuple: Single-element tuple containing the processed mask tensor
        """
        # Ensure mask is a 3D tensor
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # Add batch dimension if missing

        # Convert mask to numpy
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        
        # Ensure mask is 2D by taking the first batch if needed
        mask_np = mask_np[0]  # Take first batch
        
        # Create an empty output mask for rendering effects
        output_mask = np.zeros_like(mask_np)
        
        # Calculate centroid
        center = self.calculate_mask_centroid(mask_np)
        print(f"Centroid: {center}")
        
        # Detect edge points
        edge_points = self.detect_edge_points(mask_np, center)
        print(f"Edge Points: {len(edge_points)} found")
        
        # Calculate base line width
        base_line_width = self.calculate_base_line_width(edge_points, line_width)
        print(f"Base Line Width: {base_line_width}")
        
        # Select evenly distributed points
        selected_points = self.redistribute_points(edge_points, line_count)
        print(f"Selected Points: {len(selected_points)}")

        # Generate all effects data first
        effects_data = []
        for i in range(len(selected_points)):
            point = selected_points[i]
            next_point = selected_points[(i + 1) % len(selected_points)]
            
            # Calculate effect length based on distance from center
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            distance_from_center = np.sqrt(dx*dx + dy*dy)
            effect_length = distance_from_center * line_length
            
            effect = self.generate_flame_ray_effect(
                point, 
                next_point, 
                effect_length, 
                center,
                base_line_width
            )
            effects_data.append(effect)

        # Render all effects onto the empty output mask
        for effect in effects_data:
            output_mask = self.render_effect_to_mask(output_mask, effect)

        # Return only the rendered effects
        return (torch.from_numpy(output_mask),)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "MaskContourProcessor": MaskContourProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskContourProcessor": "Mask Contour Processor"
}
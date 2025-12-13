import random
import math
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QRadialGradient, QLinearGradient
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPolygonF


class FloatingPixelsWidget(QWidget):
    """Simple floating pixels background"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.pixels = []
        for _ in range(50):
            self.pixels.append({
                'x': random.uniform(0, 1),
                'y': random.uniform(0, 1),
                'vx': random.uniform(-0.001, 0.001),
                'vy': random.uniform(-0.001, 0.001),
                'size': random.uniform(2, 5),
            })
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)
        self.time = 0.0
    
    def animate(self):
        self.time += 0.05
        for pixel in self.pixels:
            pixel['x'] += pixel['vx']
            pixel['y'] += pixel['vy']
            if pixel['x'] < 0:
                pixel['x'] = 1
            elif pixel['x'] > 1:
                pixel['x'] = 0
            if pixel['y'] < 0:
                pixel['y'] = 1
            elif pixel['y'] > 1:
                pixel['y'] = 0
        self.update()
    
    def paintEvent(self, event):  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Qt.GlobalColor.white)
        w = self.width()
        h = self.height()
        for pixel in self.pixels:
            x = pixel['x'] * w
            y = pixel['y'] * h
            size = pixel['size']
            painter.drawEllipse(QRectF(x - size, y - size, size * 2, size * 2))


class GalaxyBackgroundWidget(QWidget):
    """Animated galaxy background with cosmic smoke and streaking particles"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # Static stars
        self.stars = []
        self.init_stars()
        
        # Nebula/smoke clouds
        self.nebulas = []
        self.init_nebulas()
        
        # Streaking particles
        self.particles = []
        self.init_particles()
        
        # Conway's Game of Life - simple original rules
        self.cell_size = 6  # Cell size in pixels
        self.grid_width = 0
        self.grid_height = 0
        self.grid = {}  # Dictionary: (x, y) -> 1 (alive) or 0 (dead)
        self.next_grid = {}
        self.game_of_life_timer = 0
        self.game_of_life_interval = 0.2  # Update every 200ms
        self.max_cells = 1000  # Limit total cells for performance
        # Viewport optimization - only process visible area
        self.visible_margin = 2  # Process cells slightly outside viewport for smooth edges
        self.viewport_grid_bounds = None  # (min_x, max_x, min_y, max_y) in grid coordinates
        # Active window optimization - only process Conway's on active window
        self.is_active_window = False  # Track if this widget's window is active
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)  # ~60 FPS for smooth animation
        
        self.time = 0.0
        
        # Initialize Game of Life grid after a short delay to ensure widget is sized
        QTimer.singleShot(100, self.init_game_of_life)
    
    def init_stars(self):
        """Initialize static stars"""
        self.stars = []
        for _ in range(200):  # More stars for better visibility
            star = {
                'x': random.uniform(0, 1),
                'y': random.uniform(0, 1),
                'size': random.uniform(0.5, 2),
                'brightness': random.uniform(0.6, 1.0),
            }
            self.stars.append(star)
        
    def init_nebulas(self):
        """Initialize cosmic smoke/nebula clouds - more gaseous with multiple layers"""
        self.nebulas = []
        for _ in range(6):  # Fewer main nebulas, but each has multiple layers
            # Create a nebula with multiple "puffs" for gaseous effect
            nebula = {
                'x': random.uniform(0, 1),
                'y': random.uniform(0, 1),
                'speed_x': random.uniform(-0.0001, 0.0001),
                'speed_y': random.uniform(-0.0001, 0.0001),
                'color_r': random.choice([120, 140, 160, 180]),  # More violet/purple tones
                'color_g': random.choice([40, 60, 80, 100]),  # Less green for more violet
                'color_b': random.choice([180, 200, 220, 240]),  # More blue for violet
                'intensity': random.uniform(0.25, 0.5),  # Slightly higher for visibility
                'puffs': []  # Multiple puffs per nebula for gaseous look
            }
            # Create 3-5 puffs per nebula with varying sizes and positions
            num_puffs = random.randint(3, 5)
            for i in range(num_puffs):
                puff = {
                    'offset_x': random.uniform(-0.15, 0.15),  # Offset from center
                    'offset_y': random.uniform(-0.15, 0.15),
                    'radius': random.uniform(0.08, 0.15),  # Size of puff
                    'intensity_mult': random.uniform(0.7, 1.0),  # Individual intensity
                }
                nebula['puffs'].append(puff)
            self.nebulas.append(nebula)
    
    def init_particles(self):
        """Initialize streaking particles - all start off-screen"""
        self.particles = []
        # Use a default screen size for initialization (will be updated on first animate)
        default_w = 1280
        default_h = 720
        
        for _ in range(15):  # Fewer but more visible
            # Random direction - can come from any side
            direction = random.choice(['left', 'right', 'top', 'bottom', 'diagonal'])
            
            # Very slow speeds
            base_speed = random.uniform(0.0008, 0.003)  # Even slower max speed
            
            # Set starting position and velocity based on direction - ALL OFF-SCREEN
            if direction == 'left':
                # Coming from left, going right - well off left edge
                start_x = random.uniform(-300, -100)
                start_y = random.uniform(-50, default_h + 50)
                angle = random.uniform(-math.pi/4, math.pi/4)
                vx = base_speed * abs(math.cos(angle))
                vy = base_speed * math.sin(angle)
            elif direction == 'right':
                # Coming from right, going left - well off right edge
                start_x = random.uniform(default_w + 100, default_w + 300)
                start_y = random.uniform(-50, default_h + 50)
                angle = random.uniform(math.pi - math.pi/4, math.pi + math.pi/4)
                vx = base_speed * math.cos(angle)
                vy = base_speed * math.sin(angle)
            elif direction == 'top':
                # Coming from top, going down - well above screen
                start_x = random.uniform(-50, default_w + 50)
                start_y = random.uniform(-300, -100)
                angle = random.uniform(math.pi/2 - math.pi/4, math.pi/2 + math.pi/4)
                vx = base_speed * math.cos(angle)
                vy = base_speed * abs(math.sin(angle))
            elif direction == 'bottom':
                # Coming from bottom, going up - well below screen
                start_x = random.uniform(-50, default_w + 50)
                start_y = random.uniform(default_h + 100, default_h + 300)
                angle = random.uniform(-math.pi/2 - math.pi/4, -math.pi/2 + math.pi/4)
                vx = base_speed * math.cos(angle)
                vy = base_speed * math.sin(angle)
            else:  # diagonal
                # Random diagonal direction - ensure completely off-screen
                start_side = random.choice(['left', 'right', 'top', 'bottom'])
                if start_side == 'left':
                    start_x = random.uniform(-300, -100)
                    start_y = random.uniform(-50, default_h + 50)
                elif start_side == 'right':
                    start_x = random.uniform(default_w + 100, default_w + 300)
                    start_y = random.uniform(-50, default_h + 50)
                elif start_side == 'top':
                    start_x = random.uniform(-50, default_w + 50)
                    start_y = random.uniform(-300, -100)
                else:  # bottom
                    start_x = random.uniform(-50, default_w + 50)
                    start_y = random.uniform(default_h + 100, default_h + 300)
                # Random diagonal angle
                angle = random.uniform(0, math.pi * 2)
                vx = base_speed * math.cos(angle)
                vy = base_speed * math.sin(angle)
            
            particle = {
                'x': start_x,
                'y': start_y,
                'vx': vx,
                'vy': vy,
                'size': random.uniform(1.5, 4.5),
                'brightness': random.uniform(0.7, 1.0),
                'flash_phase': random.uniform(0, math.pi * 2),
                'flash_speed': random.uniform(0.05, 0.25),
                'trail_length': random.uniform(60, 180),
                'spawn_delay': random.uniform(0, 8.0),  # More spawn delay variation
                'direction': direction,  # Store direction for reset
            }
            self.particles.append(particle)
    
    def init_game_of_life(self):
        """Initialize Conway's Game of Life grid"""
        w = self.width()
        h = self.height()
        if w == 0 or h == 0:
            return
        
        self.grid_width = w // self.cell_size
        self.grid_height = h // self.cell_size
        self.grid = {}
        self.next_grid = {}
        self.game_of_life_timer = 0
    
    def _calculate_viewport_bounds(self):
        """Calculate visible grid bounds based on widget size"""
        w = self.width()
        h = self.height()
        if w == 0 or h == 0:
            return None
        
        # Convert pixel bounds to grid coordinates with margin
        min_grid_x = max(0, -self.visible_margin)
        max_grid_x = min(self.grid_width, (w // self.cell_size) + self.visible_margin)
        min_grid_y = max(0, -self.visible_margin)
        max_grid_y = min(self.grid_height, (h // self.cell_size) + self.visible_margin)
        
        return (min_grid_x, max_grid_x, min_grid_y, max_grid_y)
    
    def _is_in_viewport(self, x, y):
        """Check if grid coordinates are in visible viewport"""
        if self.viewport_grid_bounds is None:
            return True  # If viewport not calculated, process everything
        min_x, max_x, min_y, max_y = self.viewport_grid_bounds
        return min_x <= x < max_x and min_y <= y < max_y
    
    def spawn_cells_from_nebula(self, nebula):
        """Spawn Game of Life cells from a nebula position - simple patterns"""
        # Don't spawn if we're at max cells
        if len(self.grid) >= self.max_cells:
            return
        
        w = self.width()
        h = self.height()
        if w == 0 or h == 0:
            return
        
        # Convert nebula position to grid coordinates
        grid_x = int(nebula['x'] * self.grid_width)
        grid_y = int(nebula['y'] * self.grid_height)
        
        # Only spawn if nebula is in or near viewport
        if not self._is_in_viewport(grid_x, grid_y):
            viewport_bounds = self._calculate_viewport_bounds()
            if viewport_bounds:
                min_x, max_x, min_y, max_y = viewport_bounds
                spawn_margin = 15  # Spawn cells slightly outside viewport
                if (grid_x < min_x - spawn_margin or grid_x > max_x + spawn_margin or
                    grid_y < min_y - spawn_margin or grid_y > max_y + spawn_margin):
                    return  # Too far from viewport, don't spawn
        
        # Reduced spawn chance for performance
        spawn_chance = 0.02  # 2% chance per nebula per update
        if random.random() < spawn_chance:
            offset_x = random.randint(-8, 8)
            offset_y = random.randint(-8, 8)
            
            center_x = grid_x + offset_x
            center_y = grid_y + offset_y
            
            # Sometimes spawn known patterns, sometimes random
            if random.random() < 0.4:  # 40% chance for known patterns
                pattern_type = random.choice(['block', 'blinker', 'glider', 'random'])
                
                if pattern_type == 'block':
                    # 2x2 block (still life)
                    pattern = [(0, 0), (1, 0), (0, 1), (1, 1)]
                elif pattern_type == 'blinker':
                    # Blinker oscillator (3 cells in a line)
                    if random.random() < 0.5:
                        pattern = [(0, 0), (1, 0), (2, 0)]  # Horizontal
                    else:
                        pattern = [(0, 0), (0, 1), (0, 2)]  # Vertical
                elif pattern_type == 'glider':
                    # Glider pattern (moves across screen)
                    pattern = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]
                else:  # random
                    pattern = None
                
                if pattern:
                    for dx, dy in pattern:
                        x = center_x + dx
                        y = center_y + dy
                        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                            # Only spawn in viewport or near it
                            if not self._is_in_viewport(x, y):
                                viewport_bounds = self._calculate_viewport_bounds()
                                if viewport_bounds:
                                    min_x, max_x, min_y, max_y = viewport_bounds
                                    if (x < min_x - 5 or x > max_x + 5 or
                                        y < min_y - 5 or y > max_y + 5):
                                        continue  # Too far, skip
                            self.grid[(x, y)] = 1  # Simple: 1 = alive
            else:
                # Random organic pattern
                pattern_size = random.randint(3, 5)
                for dx in range(-pattern_size, pattern_size + 1):
                    for dy in range(-pattern_size, pattern_size + 1):
                        if random.random() < 0.25:  # 25% chance to spawn each cell
                            x = center_x + dx
                            y = center_y + dy
                            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                                # Only spawn in viewport or near it
                                if not self._is_in_viewport(x, y):
                                    viewport_bounds = self._calculate_viewport_bounds()
                                    if viewport_bounds:
                                        min_x, max_x, min_y, max_y = viewport_bounds
                                        if (x < min_x - 5 or x > max_x + 5 or
                                            y < min_y - 5 or y > max_y + 5):
                                            continue  # Too far, skip
                                self.grid[(x, y)] = 1  # Simple: 1 = alive
    
    def update_game_of_life(self):
        """Update Conway's Game of Life one generation - simple original rules"""
        # Calculate viewport bounds for visible-only processing
        self.viewport_grid_bounds = self._calculate_viewport_bounds()
        
        # Limit cells for performance
        if len(self.grid) > self.max_cells:
            # Remove random cells to stay under limit
            keys = list(self.grid.keys())
            random.shuffle(keys)
            self.grid = {k: self.grid[k] for k in keys[:self.max_cells]}
        
        if not self.grid:
            return
        
        self.next_grid = {}
        
        # Get all cells and their neighbors - only in viewport
        cells_to_check = set()
        for (x, y) in self.grid.keys():
            if not self._is_in_viewport(x, y):
                continue  # Skip off-screen cells
            cells_to_check.add((x, y))
            # Add neighbors (only 8, not 9 including self)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if self._is_in_viewport(nx, ny):
                        cells_to_check.add((nx, ny))
        
        # Apply Conway's original rules
        for (x, y) in cells_to_check:
            if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
                continue
            
            # Count living neighbors
            neighbors = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in self.grid and self.grid[(nx, ny)] == 1:
                        neighbors += 1
            
            # Conway's original rules:
            # 1. Live cell with 2-3 neighbors survives
            # 2. Dead cell with exactly 3 neighbors becomes alive
            # 3. Everything else dies or stays dead
            is_alive = (x, y) in self.grid and self.grid[(x, y)] == 1
            
            if is_alive:
                if neighbors == 2 or neighbors == 3:
                    self.next_grid[(x, y)] = 1  # Survives
            else:
                if neighbors == 3:
                    self.next_grid[(x, y)] = 1  # Becomes alive
        
        # Clean up cells that are too far from viewport (performance)
        if self.viewport_grid_bounds:
            min_x, max_x, min_y, max_y = self.viewport_grid_bounds
            cleanup_margin = 50  # Remove cells far from viewport
            cells_to_remove = []
            for (x, y) in self.grid.keys():
                if (x < min_x - cleanup_margin or x > max_x + cleanup_margin or
                    y < min_y - cleanup_margin or y > max_y + cleanup_margin):
                    cells_to_remove.append((x, y))
            for key in cells_to_remove:
                del self.grid[key]
        
        # Update grid
        self.grid = self.next_grid
    
    def _check_if_active_window(self):
        """Check if this widget's window is the active window"""
        try:
            from PyQt5.QtWidgets import QApplication
            widget_window = self.window()
            if widget_window is None:
                return False
            
            # Check multiple ways to determine if window is active
            active_window = QApplication.activeWindow()
            if active_window is not None:
                # Direct comparison
                if widget_window == active_window:
                    return True
                # Check if it's the same window object
                if widget_window.isActiveWindow():
                    return True
                # Check if widget is in the active window's hierarchy
                if hasattr(active_window, 'centralWidget'):
                    central = active_window.centralWidget()
                    if central and self.parent() == central:
                        return True
            
            # Fallback: check if window is visible and has focus
            return widget_window.isVisible() and widget_window.hasFocus()
        except:
            # Fallback to True if check fails (safer to animate than not)
            return True
    
    def resizeEvent(self, event):  # type: ignore[override]
        """Reinitialize when resized"""
        super().resizeEvent(event)
        self.init_stars()
        self.init_nebulas()
        self.init_particles()
        self.init_game_of_life()
    
    def animate(self):
        """Update animation"""
        self.time += 0.016  # ~60 FPS
        
        w = self.width()
        h = self.height()
        
        # Check if this window is active
        self.is_active_window = self._check_if_active_window()
        
        # Initialize Game of Life grid if needed
        if self.grid_width == 0 or self.grid_height == 0:
            self.init_game_of_life()
        
        # Update Game of Life at slower interval - OPTIMIZED
        # Only update if this window is active
        self.game_of_life_timer += 0.016
        if self.game_of_life_timer >= self.game_of_life_interval:
            self.game_of_life_timer = 0
            
            # Only process Conway's Game of Life if this window is active
            if self.is_active_window:
                # Only spawn if under cell limit
                if len(self.grid) < self.max_cells:
                    # Spawn cells from nebulas (limited)
                    for nebula in self.nebulas[:3]:  # Only check first 3 nebulas for performance
                        self.spawn_cells_from_nebula(nebula)
                
                # Update Game of Life only if we have cells
                if self.grid:
                    self.update_game_of_life()
            else:
                # Window is not active - keep cells but don't process them
                # Optionally reduce cell count slightly to save memory
                if len(self.grid) > 200:
                    # Keep a subset of cells for when window becomes active again
                    # Keep cells that are in or near viewport
                    viewport_bounds = self._calculate_viewport_bounds()
                    if viewport_bounds:
                        min_x, max_x, min_y, max_y = viewport_bounds
                        margin = 20
                        cells_to_keep = {}
                        for (x, y), cell_value in self.grid.items():
                            if (min_x - margin <= x < max_x + margin and 
                                min_y - margin <= y < max_y + margin):
                                cells_to_keep[(x, y)] = cell_value
                        # Keep at least 100 cells
                        if len(cells_to_keep) < 100:
                            cells_to_keep = dict(list(self.grid.items())[:100])
                        self.grid = cells_to_keep
        
        # Animate nebulas
        nebula_moved = False
        for nebula in self.nebulas:
            old_x, old_y = nebula['x'], nebula['y']
            nebula['x'] += nebula['speed_x']
            nebula['y'] += nebula['speed_y']
            
            # Wrap around edges
            if nebula['x'] < -0.5:
                nebula['x'] = 1.5
                nebula_moved = True
            elif nebula['x'] > 1.5:
                nebula['x'] = -0.5
                nebula_moved = True
            if nebula['y'] < -0.5:
                nebula['y'] = 1.5
                nebula_moved = True
            elif nebula['y'] > 1.5:
                nebula['y'] = -0.5
                nebula_moved = True
            
            # Check if nebula actually moved
            if abs(nebula['x'] - old_x) > 0.001 or abs(nebula['y'] - old_y) > 0.001:
                nebula_moved = True
        
        # Animate particles (shooting stars) - all use pixel coordinates
        for particle in self.particles:
            # Handle spawn delay
            if particle.get('spawn_delay', 0) > 0:
                particle['spawn_delay'] -= 0.016
                continue
            
            # All particles now use pixel coordinates directly
            # Update position (vx and vy are normalized, scale by screen size)
            particle['x'] += particle['vx'] * w
            particle['y'] += particle['vy'] * h
            
            # Reset when off screen - ensure ALL start well off-screen
            if particle['x'] > w + 200 or particle['x'] < -200 or particle['y'] > h + 200 or particle['y'] < -200:
                # Random direction - can come from any side
                direction = random.choice(['left', 'right', 'top', 'bottom', 'diagonal'])
                
                # Very slow speeds
                base_speed = random.uniform(0.0008, 0.003)  # Even slower max speed
                
                # Set starting position and velocity based on direction - ALL WELL OFF-SCREEN
                if direction == 'left':
                    # Coming from left, going right - well off left edge
                    particle['x'] = random.uniform(-300, -100)
                    particle['y'] = random.uniform(-50, h + 50)
                    angle = random.uniform(-math.pi/4, math.pi/4)
                    particle['vx'] = base_speed * abs(math.cos(angle))
                    particle['vy'] = base_speed * math.sin(angle)
                elif direction == 'right':
                    # Coming from right, going left - well off right edge
                    particle['x'] = random.uniform(w + 100, w + 300)
                    particle['y'] = random.uniform(-50, h + 50)
                    angle = random.uniform(math.pi - math.pi/4, math.pi + math.pi/4)
                    particle['vx'] = base_speed * math.cos(angle)
                    particle['vy'] = base_speed * math.sin(angle)
                elif direction == 'top':
                    # Coming from top, going down - well above screen
                    particle['x'] = random.uniform(-50, w + 50)
                    particle['y'] = random.uniform(-300, -100)
                    angle = random.uniform(math.pi/2 - math.pi/4, math.pi/2 + math.pi/4)
                    particle['vx'] = base_speed * math.cos(angle)
                    particle['vy'] = base_speed * abs(math.sin(angle))
                elif direction == 'bottom':
                    # Coming from bottom, going up - well below screen
                    particle['x'] = random.uniform(-50, w + 50)
                    particle['y'] = random.uniform(h + 100, h + 300)
                    angle = random.uniform(-math.pi/2 - math.pi/4, -math.pi/2 + math.pi/4)
                    particle['vx'] = base_speed * math.cos(angle)
                    particle['vy'] = base_speed * math.sin(angle)
                else:  # diagonal
                    # Random diagonal direction - ensure completely off-screen
                    start_side = random.choice(['left', 'right', 'top', 'bottom'])
                    if start_side == 'left':
                        particle['x'] = random.uniform(-300, -100)
                        particle['y'] = random.uniform(-50, h + 50)
                    elif start_side == 'right':
                        particle['x'] = random.uniform(w + 100, w + 300)
                        particle['y'] = random.uniform(-50, h + 50)
                    elif start_side == 'top':
                        particle['x'] = random.uniform(-50, w + 50)
                        particle['y'] = random.uniform(-300, -100)
                    else:  # bottom
                        particle['x'] = random.uniform(-50, w + 50)
                        particle['y'] = random.uniform(h + 100, h + 300)
                    # Random diagonal angle
                    angle = random.uniform(0, math.pi * 2)
                    particle['vx'] = base_speed * math.cos(angle)
                    particle['vy'] = base_speed * math.sin(angle)
                
                # More random properties
                particle['size'] = random.uniform(1.5, 4.5)
                particle['brightness'] = random.uniform(0.7, 1.0)
                particle['flash_phase'] = random.uniform(0, math.pi * 2)
                particle['flash_speed'] = random.uniform(0.05, 0.25)
                particle['trail_length'] = random.uniform(60, 180)
                particle['spawn_delay'] = random.uniform(0, 8.0)
                particle['direction'] = direction
        
        self.update()
    
    def paintEvent(self, event):  # type: ignore[override]
        """Paint the galaxy background"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Base - fully transparent so stars show through
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))  # Completely transparent
        
        # Draw stars first (so they're behind everything and always visible)
        painter.setPen(Qt.PenStyle.NoPen)
        for star in self.stars:
            star_x = star['x'] * w
            star_y = star['y'] * h
            star_size = star['size']
            star_brightness = int(255 * star['brightness'])
            
            # Draw star as a small bright dot
            star_color = QColor(255, 255, 255, star_brightness)
            painter.setBrush(star_color)
            painter.drawEllipse(QRectF(star_x - star_size, star_y - star_size, star_size * 2, star_size * 2))
        
        # Draw nebulas (cosmic smoke) - gaseous effect with multiple puffs
        for nebula in self.nebulas:
            base_x = nebula['x'] * w
            base_y = nebula['y'] * h
            
            # Draw each puff in the nebula for gaseous effect
            for puff in nebula['puffs']:
                # Calculate puff position with offset
                puff_x = base_x + (puff['offset_x'] * w)
                puff_y = base_y + (puff['offset_y'] * h)
                puff_radius = puff['radius'] * max(w, h)
                puff_intensity = nebula['intensity'] * puff['intensity_mult']
                
                # Create radial gradient for each puff - more organic gaseous look
                gradient = QRadialGradient(puff_x, puff_y, puff_radius)
                
                # Center is brighter, edges fade to transparent
                center_color = QColor(
                    int(nebula['color_r'] * puff_intensity),
                    int(nebula['color_g'] * puff_intensity),
                    int(nebula['color_b'] * puff_intensity),
                    int(60 * puff_intensity)  # Moderate opacity for visibility
                )
                edge_color = QColor(0, 0, 0, 0)  # Fully transparent edge
                
                # Multiple gradient stops for smooth gaseous fade
                gradient.setColorAt(0.0, center_color)
                gradient.setColorAt(0.3, QColor(
                    int(nebula['color_r'] * puff_intensity * 0.85),
                    int(nebula['color_g'] * puff_intensity * 0.85),
                    int(nebula['color_b'] * puff_intensity * 0.85),
                    int(45 * puff_intensity)
                ))
                gradient.setColorAt(0.5, QColor(
                    int(nebula['color_r'] * puff_intensity * 0.65),
                    int(nebula['color_g'] * puff_intensity * 0.65),
                    int(nebula['color_b'] * puff_intensity * 0.65),
                    int(30 * puff_intensity)
                ))
                gradient.setColorAt(0.7, QColor(
                    int(nebula['color_r'] * puff_intensity * 0.4),
                    int(nebula['color_g'] * puff_intensity * 0.4),
                    int(nebula['color_b'] * puff_intensity * 0.4),
                    int(15 * puff_intensity)
                ))
                gradient.setColorAt(0.85, QColor(
                    int(nebula['color_r'] * puff_intensity * 0.2),
                    int(nebula['color_g'] * puff_intensity * 0.2),
                    int(nebula['color_b'] * puff_intensity * 0.2),
                    int(5 * puff_intensity)
                ))
                gradient.setColorAt(1.0, edge_color)
                
                painter.setBrush(gradient)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QRectF(puff_x - puff_radius, puff_y - puff_radius, 
                                          puff_radius * 2, puff_radius * 2))
        
        # Draw Conway's Game of Life cells - simple rendering (only visible and active window)
        # Only render if this window is active
        if self.is_active_window:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setRenderHint(QPainter.Antialiasing, False)  # Pixel art style
            
            # Only render cells in visible viewport
            viewport_bounds = self._calculate_viewport_bounds()
            for (grid_x, grid_y), cell_value in self.grid.items():
                # Skip off-screen cells
                if viewport_bounds:
                    min_x, max_x, min_y, max_y = viewport_bounds
                    if not (min_x <= grid_x < max_x and min_y <= grid_y < max_y):
                        continue
                
                # Only draw alive cells
                if cell_value != 1:
                    continue
                
                # Convert grid coordinates to pixel coordinates
                pixel_x = grid_x * self.cell_size
                pixel_y = grid_y * self.cell_size
                
                # Simple purple/violet color for cells
                cell_color = QColor(140, 80, 200, 200)  # Purple with good visibility
                painter.setBrush(cell_color)
                
                # Draw simple square cells (pixel art style)
                painter.drawRect(pixel_x, pixel_y, self.cell_size, self.cell_size)
        
        # Re-enable antialiasing for other elements
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Draw streaking particles (shooting stars) - much more visible
        for particle in self.particles:
            # Skip if still in spawn delay
            if particle.get('spawn_delay', 0) > 0:
                continue
            
            # Only draw if on screen or close to it
            if particle['x'] < -200 or particle['x'] > w + 200:
                continue
            if particle['y'] < -200 or particle['y'] > h + 200:
                continue
            
            # Calculate flash brightness
            flash = 0.7 + 0.3 * math.sin(self.time * particle['flash_speed'] + particle['flash_phase'])
            brightness = particle['brightness'] * flash
            
            # Draw trail - much longer and brighter
            trail_length = particle['trail_length']
            # Calculate velocity direction in pixels
            # vx and vy are normalized speeds (0-1 range), need to scale by screen size
            vel_x_px = particle['vx'] * w
            vel_y_px = particle['vy'] * h
            vel_mag = math.sqrt(vel_x_px ** 2 + vel_y_px ** 2)
            
            if vel_mag > 0:
                # Calculate trail start point (behind the particle in direction of movement)
                # Normalize velocity vector and scale by trail_length
                vel_norm_x = vel_x_px / vel_mag
                vel_norm_y = vel_y_px / vel_mag
                trail_x = particle['x'] - (vel_norm_x * trail_length)
                trail_y = particle['y'] - (vel_norm_y * trail_length)
            else:
                # Fallback if no velocity
                trail_x = particle['x'] - trail_length
                trail_y = particle['y']
            
            # Create gradient for trail - fades from tail end to bright head
            gradient = QLinearGradient(trail_x, trail_y, particle['x'], particle['y'])
            # Fade from transparent at tail end to bright at head
            gradient.setColorAt(0.0, QColor(200, 150, 255, 0))  # Fully transparent at tail end
            gradient.setColorAt(0.3, QColor(200, 150, 255, int(40 * brightness)))  # Start appearing
            gradient.setColorAt(0.5, QColor(220, 180, 255, int(80 * brightness)))  # Getting brighter
            gradient.setColorAt(0.7, QColor(240, 220, 255, int(140 * brightness)))  # Brighter violet
            gradient.setColorAt(0.85, QColor(255, 240, 255, int(180 * brightness)))  # Almost white
            gradient.setColorAt(1.0, QColor(255, 255, 255, int(220 * brightness)))  # Bright white at head
            
            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            
            # Draw trail as a line along the direction of movement
            trail_width = 2
            # Calculate perpendicular vector for trail width
            if vel_mag > 0:
                perp_x = -vel_norm_y * trail_width / 2
                perp_y = vel_norm_x * trail_width / 2
            else:
                perp_x = 0
                perp_y = trail_width / 2
            
            # Create a polygon for the trail (rectangle aligned with movement)
            from PyQt5.QtCore import QPointF
            from PyQt5.QtGui import QPolygonF
            trail_points = QPolygonF([
                QPointF(trail_x + perp_x, trail_y + perp_y),
                QPointF(trail_x - perp_x, trail_y - perp_y),
                QPointF(particle['x'] - perp_x, particle['y'] - perp_y),
                QPointF(particle['x'] + perp_x, particle['y'] + perp_y)
            ])
            painter.drawPolygon(trail_points)
            
            # Draw head of shooting star - bright white/violet
            head_size = particle['size']
            head_color = QColor(255, 255, 255, int(255 * brightness))
            painter.setBrush(head_color)
            painter.drawEllipse(QRectF(
                particle['x'] - head_size,
                particle['y'] - head_size,
                head_size * 2,
                head_size * 2
            ))

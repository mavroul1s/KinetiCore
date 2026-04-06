import cv2
import mediapipe as mp
import numpy as np

class InteractiveMolecule:
    def __init__(self):
        # 1. Setup MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 2. Define Molecule (Methane)
        self.atoms = np.array([
            [0, 0, 0],          # 0: Carbon
            [1, 1, 1],          # 1: H
            [1, -1, -1],        # 2: H
            [-1, 1, -1],        # 3: H
            [-1, -1, 1]         # 4: H
        ], dtype=float) * 120   

        self.bonds = [[0, 1], [0, 2], [0, 3], [0, 4]]
        
        # Slightly adjusted colors for a better look
        self.colors = [
            (80, 80, 80),       # Carbon
            (220, 220, 220),    # H
            (220, 220, 220),    # H
            (220, 220, 220),    # H
            (220, 220, 220)     # H
        ]

        # 3. State Variables
        self.angle_x = 0
        self.angle_y = 0
        
        # Interaction States
        self.last_cursor = None     # Where was the hand last frame?
        self.selected_atom = None   # Which atom are we pinching?
        self.is_pinching = False    # Are fingers touching?

    def get_rotation_matrices(self):
        """Returns the current rotation matrices."""
        cos_x, sin_x = np.cos(self.angle_x), np.sin(self.angle_x)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])

        cos_y, sin_y = np.cos(self.angle_y), np.sin(self.angle_y)
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        
        return R_x, R_y

    def rotate_points(self, points):
        """Rotates points for display."""
        R_x, R_y = self.get_rotation_matrices()
        rotated = np.dot(points, R_x)
        rotated = np.dot(rotated, R_y)
        return rotated

    def inverse_rotate_vector(self, vec_2d):
        """Calculates 3D movement from 2D screen movement."""
        dx, dy = vec_2d
        vec_view = np.array([dx, dy, 0])
        R_x, R_y = self.get_rotation_matrices()
        
        inv_vec = np.dot(vec_view, R_y.T)
        inv_vec = np.dot(inv_vec, R_x.T)
        return inv_vec

    def project_points(self, points, width, height):
        projected = []
        for point in points:
            x, y, z = point
            factor = 600 / (z + 600) 
            x_2d = int(x * factor + width / 2)
            y_2d = int(y * factor + height / 2)
            projected.append((x_2d, y_2d))
        return projected

    def run(self):
        cap = cv2.VideoCapture(0)
        MOL_W, MOL_H = 800, 600

        print("controls:")
        print(" - Open Hand + Move: Rotate Molecule")
        print(" - Pinch (Index+Thumb) on Atom: Grab and Stretch")
        
        while cap.isOpened():
            success, cam_view = cap.read()
            if not success: break

            cam_view = cv2.flip(cam_view, 1)
            h_cam, w_cam, _ = cam_view.shape
            
            # Create Black Window for Molecule
            mol_view = np.zeros((MOL_H, MOL_W, 3), dtype=np.uint8)

            # --- Draw Background Grid ---
            grid_color = (30, 40, 50) # Dark blue/grey grid
            grid_spacing = 40
            for x in range(0, MOL_W, grid_spacing):
                cv2.line(mol_view, (x, 0), (x, MOL_H), grid_color, 1)
            for y in range(0, MOL_H, grid_spacing):
                cv2.line(mol_view, (0, y), (MOL_W, y), grid_color, 1)

            img_rgb = cv2.cvtColor(cam_view, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            cursor_norm_x, cursor_norm_y = None, None
            
            # --- Hand Logic ---
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(cam_view, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                    index_tip = hand_lms.landmark[8]
                    thumb_tip = hand_lms.landmark[4]

                    cursor_norm_x = index_tip.x
                    cursor_norm_y = index_tip.y

                    pinch_dist = np.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                    self.is_pinching = pinch_dist < 0.05

                    color = (0, 255, 0) if self.is_pinching else (0, 0, 255)
                    ix, iy = int(index_tip.x * w_cam), int(index_tip.y * h_cam)
                    tx, ty = int(thumb_tip.x * w_cam), int(thumb_tip.y * h_cam)
                    cv2.line(cam_view, (ix, iy), (tx, ty), color, 3)

            # --- Interaction Logic ---
            cursor_pos = None
            if cursor_norm_x is not None:
                cursor_x = int(cursor_norm_x * MOL_W)
                cursor_y = int(cursor_norm_y * MOL_H)
                cursor_pos = (cursor_x, cursor_y)

                cursor_color = (0, 255, 0) if self.is_pinching else (0, 255, 255)
                cv2.circle(mol_view, cursor_pos, 10, cursor_color, 2, cv2.LINE_AA)
                cv2.line(mol_view, (cursor_x - 15, cursor_y), (cursor_x + 15, cursor_y), cursor_color, 1, cv2.LINE_AA)
                cv2.line(mol_view, (cursor_x, cursor_y - 15), (cursor_x, cursor_y + 15), cursor_color, 1, cv2.LINE_AA)

            # 3D CALCULATIONS
            rotated_atoms = self.rotate_points(self.atoms)
            projected_atoms = self.project_points(rotated_atoms, MOL_W, MOL_H)

            # Interaction Handling
            if cursor_pos:
                if self.is_pinching:
                    if self.selected_atom is None:
                        closest_idx = -1
                        min_dist = 40 
                        
                        for i, pt in enumerate(projected_atoms):
                            dist = np.hypot(cursor_pos[0] - pt[0], cursor_pos[1] - pt[1])
                            if dist < min_dist:
                                min_dist = dist
                                closest_idx = i
                        
                        if closest_idx != -1:
                            self.selected_atom = closest_idx
                    
                    if self.selected_atom is not None and self.last_cursor is not None:
                        dx = cursor_pos[0] - self.last_cursor[0]
                        dy = cursor_pos[1] - self.last_cursor[1]
                        
                        move_vec_3d = self.inverse_rotate_vector((dx, dy))
                        self.atoms[self.selected_atom] += move_vec_3d

                else:
                    self.selected_atom = None 
                    
                    if self.last_cursor is not None:
                        dx = cursor_pos[0] - self.last_cursor[0]
                        dy = cursor_pos[1] - self.last_cursor[1]
                        self.angle_y += dx * 0.005
                        self.angle_x -= dy * 0.005

                self.last_cursor = cursor_pos
            else:
                self.last_cursor = None
                self.selected_atom = None

            # --- Render Molecule ---
            sorted_indices = np.argsort(rotated_atoms[:, 2])

            # Draw Bonds
            for p1_idx, p2_idx in self.bonds:
                pt1 = projected_atoms[p1_idx]
                pt2 = projected_atoms[p2_idx]
                # Thicker, anti-aliased bonds
                cv2.line(mol_view, pt1, pt2, (150, 150, 150), 6, cv2.LINE_AA)
                cv2.line(mol_view, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA) # Core line for detail

            # Draw Atoms
            for idx in sorted_indices:
                pt = projected_atoms[idx]
                radius = 30
                color = self.colors[idx]
                
                if idx == self.selected_atom:
                    color = (0, 255, 100) # Neon Green
                    radius = 35

                # Outer Shadow / Border
                cv2.circle(mol_view, pt, radius + 2, (50, 50, 50), -1, cv2.LINE_AA)
                
                # Main Body
                cv2.circle(mol_view, pt, radius, color, -1, cv2.LINE_AA)
                
                # Specular Highlight (Fake 3D lighting effect)
                highlight_pt = (pt[0] - int(radius * 0.3), pt[1] - int(radius * 0.3))
                highlight_radius = int(radius * 0.25)
                cv2.circle(mol_view, highlight_pt, highlight_radius, (255, 255, 255), -1, cv2.LINE_AA)

            cv2.imshow('Controller', cam_view)
            cv2.imshow('angel', mol_view)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = InteractiveMolecule()
    app.run()
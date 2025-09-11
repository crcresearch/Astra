import os
import sys
import numpy as np
import json
import cv2
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Constants
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

class PoseFeatureExtractor:
    """
    Class to extract features from pose data
    """
    def __init__(self, num_keypoints=25):
        """
        Initialize the feature extractor
        
        Args:
            num_keypoints: Number of keypoints in the pose data
        """
        self.num_keypoints = num_keypoints
        
        # Define joint relationships for angle calculation
        # Format: (joint_idx, parent_idx, child_idx)
        self.joint_triplets = [
            # Right arm
            (2, 1, 3),    # Right shoulder (neck, right elbow)
            (3, 2, 4),    # Right elbow (right shoulder, right wrist)
            # Left arm
            (5, 1, 6),    # Left shoulder (neck, left elbow)
            (6, 5, 7),    # Left elbow (left shoulder, left wrist)
            # Torso
            (1, 0, 8),    # Neck (nose, mid hip)
            # Right leg
            (9, 8, 10),   # Right hip (mid hip, right knee)
            (10, 9, 11),  # Right knee (right hip, right ankle)
            # Left leg
            (12, 8, 13),  # Left hip (mid hip, left knee)
            (13, 12, 14)  # Left knee (left hip, left ankle)
        ]
        
        # Define keypoint pairs for distance calculation
        self.keypoint_pairs = [
            # Arms span
            (4, 7),       # Right wrist to left wrist
            # Vertical distances
            (0, 8),       # Nose to mid hip (torso length)
            (8, 11),      # Mid hip to right ankle
            (8, 14),      # Mid hip to left ankle
            # Symmetry pairs
            (2, 5),       # Right shoulder to left shoulder
            (3, 6),       # Right elbow to left elbow
            (4, 7),       # Right wrist to left wrist
            (9, 12),      # Right hip to left hip
            (10, 13),     # Right knee to left knee
            (11, 14)      # Right ankle to left ankle
        ]
    
    def extract_features_from_json_files(self, json_dir):
        """
        Extract features from JSON files in a directory
        
        Args:
            json_dir: Directory containing JSON files
            
        Returns:
            Extracted features
        """
        if not os.path.exists(json_dir):
            print(f"Error: Directory does not exist: {json_dir}")
            return None
        
        # Find all JSON files in the directory
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        
        if not json_files:
            print(f"No JSON files found in {json_dir}")
            return None
        
        # Sort files by frame number
        json_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f.split('_')[-1]))))
        
        # Load pose data from each JSON file
        pose_frames = []
        
        for json_file in json_files:
            file_path = os.path.join(json_dir, json_file)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract keypoints based on the expected format
                # The format might vary, so we need to handle different cases
                keypoints = None
                
                if 'people' in data and len(data['people']) > 0:
                    # OpenPose format
                    if 'pose_keypoints_2d' in data['people'][0]:
                        flat_keypoints = data['people'][0]['pose_keypoints_2d']
                        keypoints = np.array(flat_keypoints).reshape(-1, 3)  # reshape to [keypoints, 3]
                
                if keypoints is None:
                    # Try alternative format
                    if isinstance(data, dict) and any(k.isdigit() for k in data.keys()):
                        # Format like {"0": {"x": 100, "y": 200, "confidence": 0.9}, ...}
                        keypoints = np.zeros((self.num_keypoints, 3))
                        for i in range(self.num_keypoints):
                            if str(i) in data:
                                point = data[str(i)]
                                keypoints[i, 0] = point.get('x', 0)
                                keypoints[i, 1] = point.get('y', 0)
                                keypoints[i, 2] = point.get('confidence', 0)
                
                if keypoints is not None:
                    pose_frames.append(keypoints)
            
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        if not pose_frames:
            print(f"No valid pose data found in {json_dir}")
            return None
        
        # Convert to numpy array
        pose_frames = np.array(pose_frames)
        
        # Extract features from the sequence of poses
        features = self.extract_features_from_pose_sequence(pose_frames)
        
        return features
    
    def extract_features_from_pose_sequence(self, pose_frames):
        """
        Extract features from a sequence of pose frames
        
        Args:
            pose_frames: Array of pose keypoints [frames, keypoints, coords]
            
        Returns:
            Extracted features
        """
        features = []
        
        try:
            # 1. Postural features (from mean pose)
            mean_pose = np.mean(pose_frames, axis=0)
            valid_mean_pose = mean_pose.copy()
            
            # Replace zero confidence points with NaN
            valid_mean_pose[mean_pose[:, 2] < 0.1, :] = np.nan
            
            # 1.1 Joint angles
            angle_features = self._calculate_joint_angles(valid_mean_pose)
            features.extend(angle_features)
            
            # 1.2 Body proportions
            proportion_features = self._calculate_body_proportions(valid_mean_pose)
            features.extend(proportion_features)
            
            # 1.3 Keypoint distances
            distance_features = self._calculate_keypoint_distances(valid_mean_pose)
            features.extend(distance_features)
            
            # 1.4 Symmetry features
            symmetry_features = self._calculate_symmetry_features(valid_mean_pose)
            features.extend(symmetry_features)
            
            # 2. Kinematic features (from pose sequence)
            if pose_frames.shape[0] > 1:
                # 2.1 Velocity features
                velocity_features = self._calculate_velocity_features(pose_frames)
                features.extend(velocity_features)
                
                # 2.2 Acceleration features
                if pose_frames.shape[0] > 2:
                    acceleration_features = self._calculate_acceleration_features(pose_frames)
                    features.extend(acceleration_features)
                
                # 2.3 Movement distribution features
                movement_features = self._calculate_movement_distribution(pose_frames)
                features.extend(movement_features)
                
                # 2.4 Periodicity features
                periodicity_features = self._calculate_periodicity_features(pose_frames)
                features.extend(periodicity_features)
            
            # 3. Global features
            # 3.1 Bounding box features
            bbox_features = self._calculate_bounding_box_features(pose_frames)
            features.extend(bbox_features)
            
            # 3.2 Overall movement amount
            overall_movement = self._calculate_overall_movement(pose_frames)
            features.append(overall_movement)
            
            return np.array(features)
        
        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_joint_angles(self, pose):
        """
        Calculate angles at joints
        
        Args:
            pose: Mean pose keypoints
            
        Returns:
            List of joint angles
        """
        angles = []
        
        for joint, parent, child in self.joint_triplets:
            try:
                # Get the joint, parent, and child coordinates
                joint_pos = pose[joint, :2]
                parent_pos = pose[parent, :2]
                child_pos = pose[child, :2]
                
                # Skip if any of the points have NaN values
                if np.isnan(joint_pos).any() or np.isnan(parent_pos).any() or np.isnan(child_pos).any():
                    angles.append(0)  # Default value
                    continue
                
                # Calculate vectors
                v1 = parent_pos - joint_pos
                v2 = child_pos - joint_pos
                
                # Normalize vectors
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm == 0 or v2_norm == 0:
                    angles.append(0)  # Default value
                    continue
                
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # Calculate angle using dot product
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                # Convert to degrees
                angle_deg = np.degrees(angle)
                angles.append(angle_deg)
            
            except Exception as e:
                angles.append(0)  # Default value
        
        return angles
    
    def _calculate_body_proportions(self, pose):
        """
        Calculate body proportions
        
        Args:
            pose: Mean pose keypoints
            
        Returns:
            List of body proportion features
        """
        proportions = []
        
        try:
            # Height (vertical distance from head to feet)
            head_y = pose[0, 1]  # Nose y-coordinate
            feet_y = np.nanmin([pose[11, 1], pose[14, 1]])  # Minimum of right and left ankle y-coordinates
            
            if not np.isnan(head_y) and not np.isnan(feet_y):
                height = abs(feet_y - head_y)
                proportions.append(height)
            else:
                proportions.append(0)
            
            # Width (distance between shoulders)
            left_shoulder_x = pose[5, 0]
            right_shoulder_x = pose[2, 0]
            
            if not np.isnan(left_shoulder_x) and not np.isnan(right_shoulder_x):
                width = abs(left_shoulder_x - right_shoulder_x)
                proportions.append(width)
            else:
                proportions.append(0)
            
            # Aspect ratio (height / width)
            if proportions[0] > 0 and proportions[1] > 0:
                aspect_ratio = proportions[0] / proportions[1]
                proportions.append(aspect_ratio)
            else:
                proportions.append(0)
        
        except Exception as e:
            print(f"Error calculating body proportions: {e}")
            proportions = [0, 0, 0]  # Default values
        
        return proportions
    
    def _calculate_keypoint_distances(self, pose):
        """
        Calculate distances between keypoint pairs
        
        Args:
            pose: Mean pose keypoints
            
        Returns:
            List of distance features
        """
        distances = []
        
        for idx1, idx2 in self.keypoint_pairs:
            try:
                point1 = pose[idx1, :2]
                point2 = pose[idx2, :2]
                
                if np.isnan(point1).any() or np.isnan(point2).any():
                    distances.append(0)  # Default value
                    continue
                
                distance = np.linalg.norm(point1 - point2)
                distances.append(distance)
            
            except Exception as e:
                distances.append(0)  # Default value
        
        return distances
    
    def _calculate_symmetry_features(self, pose):
        """
        Calculate body symmetry features
        
        Args:
            pose: Mean pose keypoints
            
        Returns:
            List of symmetry features
        """
        symmetry_features = []
        
        try:
            # Symmetry pairs: right-left
            symmetry_pairs = [
                (2, 5),    # Shoulders
                (3, 6),    # Elbows
                (4, 7),    # Wrists
                (9, 12),   # Hips
                (10, 13),  # Knees
                (11, 14)   # Ankles
            ]
            
            # Assume the vertical midline is at the neck (keypoint 1)
            midline_x = pose[1, 0]
            
            for right_idx, left_idx in symmetry_pairs:
                right_point = pose[right_idx, :2]
                left_point = pose[left_idx, :2]
                
                if np.isnan(right_point).any() or np.isnan(left_point).any() or np.isnan(midline_x):
                    symmetry_features.append(0)  # Default value
                    continue
                
                # Mirror the right point across the midline
                mirrored_right_x = 2 * midline_x - right_point[0]
                mirrored_right_point = np.array([mirrored_right_x, right_point[1]])
                
                # Calculate distance between mirrored right point and left point
                symmetry_distance = np.linalg.norm(mirrored_right_point - left_point)
                symmetry_features.append(symmetry_distance)
            
            # Overall symmetry score (mean of all symmetry distances)
            if symmetry_features:
                overall_symmetry = np.mean(symmetry_features)
                symmetry_features.append(overall_symmetry)
            else:
                symmetry_features.append(0)
        
        except Exception as e:
            print(f"Error calculating symmetry features: {e}")
            symmetry_features = [0] * 7  # Default values (6 pairs + overall)
        
        return symmetry_features
    
    def _calculate_velocity_features(self, pose_frames):
        """
        Calculate velocity features
        
        Args:
            pose_frames: Sequence of pose keypoints
            
        Returns:
            List of velocity features
        """
        velocity_features = []
        
        try:
            # Calculate velocities for each keypoint
            velocities = np.zeros((pose_frames.shape[0] - 1, pose_frames.shape[1], 2))
            
            for i in range(pose_frames.shape[0] - 1):
                # Calculate displacement between consecutive frames
                curr_valid = pose_frames[i, :, 2] > 0.1
                next_valid = pose_frames[i + 1, :, 2] > 0.1
                valid_mask = curr_valid & next_valid
                
                # Only calculate velocity for valid keypoints
                velocities[i, valid_mask, :] = pose_frames[i + 1, valid_mask, :2] - pose_frames[i, valid_mask, :2]
            
            # Set invalid velocities to NaN
            invalid_mask = np.isnan(velocities).any(axis=2)
            velocities[invalid_mask] = np.nan
            
            # Calculate mean and max velocity for key joints
            key_joints = [0, 1, 2, 3, 4, 5, 6, 7]  # Head, neck, shoulders, elbows, wrists
            
            for joint in key_joints:
                joint_velocities = velocities[:, joint, :]
                joint_speeds = np.linalg.norm(joint_velocities, axis=1)
                
                # Mean velocity
                mean_speed = np.nanmean(joint_speeds) if not np.isnan(joint_speeds).all() else 0
                velocity_features.append(mean_speed)
                
                # Max velocity
                max_speed = np.nanmax(joint_speeds) if not np.isnan(joint_speeds).all() else 0
                velocity_features.append(max_speed)
            
            # Overall body velocity (average of all joints)
            all_speeds = np.linalg.norm(velocities, axis=2)
            mean_overall_speed = np.nanmean(all_speeds) if not np.isnan(all_speeds).all() else 0
            velocity_features.append(mean_overall_speed)
            
            max_overall_speed = np.nanmax(all_speeds) if not np.isnan(all_speeds).all() else 0
            velocity_features.append(max_overall_speed)
        
        except Exception as e:
            print(f"Error calculating velocity features: {e}")
            velocity_features = [0] * 18  # Default values (8 joints * 2 features + 2 overall)
        
        return velocity_features
    
    def _calculate_acceleration_features(self, pose_frames):
        """
        Calculate acceleration features
        
        Args:
            pose_frames: Sequence of pose keypoints
            
        Returns:
            List of acceleration features
        """
        acceleration_features = []
        
        try:
            # Calculate velocities
            velocities = np.zeros((pose_frames.shape[0] - 1, pose_frames.shape[1], 2))
            
            for i in range(pose_frames.shape[0] - 1):
                curr_valid = pose_frames[i, :, 2] > 0.1
                next_valid = pose_frames[i + 1, :, 2] > 0.1
                valid_mask = curr_valid & next_valid
                
                velocities[i, valid_mask, :] = pose_frames[i + 1, valid_mask, :2] - pose_frames[i, valid_mask, :2]
            
            # Calculate accelerations
            accelerations = np.zeros((pose_frames.shape[0] - 2, pose_frames.shape[1], 2))
            
            for i in range(velocities.shape[0] - 1):
                curr_valid = ~np.isnan(velocities[i]).any(axis=1)
                next_valid = ~np.isnan(velocities[i + 1]).any(axis=1)
                valid_mask = curr_valid & next_valid
                
                accelerations[i, valid_mask, :] = velocities[i + 1, valid_mask, :] - velocities[i, valid_mask, :]
            
            # Key joints for acceleration features
            key_joints = [0, 4, 7]  # Head, right wrist, left wrist
            
            for joint in key_joints:
                joint_accelerations = accelerations[:, joint, :]
                joint_acc_magnitudes = np.linalg.norm(joint_accelerations, axis=1)
                
                mean_acc = np.nanmean(joint_acc_magnitudes) if not np.isnan(joint_acc_magnitudes).all() else 0
                acceleration_features.append(mean_acc)
                
                max_acc = np.nanmax(joint_acc_magnitudes) if not np.isnan(joint_acc_magnitudes).all() else 0
                acceleration_features.append(max_acc)
            
            # Overall body acceleration
            all_acc_magnitudes = np.linalg.norm(accelerations, axis=2)
            mean_overall_acc = np.nanmean(all_acc_magnitudes) if not np.isnan(all_acc_magnitudes).all() else 0
            acceleration_features.append(mean_overall_acc)
            
            max_overall_acc = np.nanmax(all_acc_magnitudes) if not np.isnan(all_acc_magnitudes).all() else 0
            acceleration_features.append(max_overall_acc)
        
        except Exception as e:
            print(f"Error calculating acceleration features: {e}")
            acceleration_features = [0] * 8  # Default values (3 joints * 2 features + 2 overall)
        
        return acceleration_features
    
    def _calculate_movement_distribution(self, pose_frames):
        """
        Calculate movement distribution features
        
        Args:
            pose_frames: Sequence of pose keypoints
            
        Returns:
            List of movement distribution features
        """
        movement_features = []
        
        try:
            # Calculate displacements between consecutive frames
            displacements = np.zeros((pose_frames.shape[0] - 1, pose_frames.shape[1]))
            
            for i in range(pose_frames.shape[0] - 1):
                curr_frame = pose_frames[i, :, :2]
                next_frame = pose_frames[i + 1, :, :2]
                
                curr_valid = pose_frames[i, :, 2] > 0.1
                next_valid = pose_frames[i + 1, :, 2] > 0.1
                valid_mask = curr_valid & next_valid
                
                # Calculate Euclidean distance for valid keypoints
                for j in range(pose_frames.shape[1]):
                    if valid_mask[j]:
                        displacements[i, j] = np.linalg.norm(next_frame[j] - curr_frame[j])
            
            # Group joints into body parts
            body_parts = {
                'head': [0, 15, 16, 17, 18],  # Nose, eyes, ears
                'arms': [2, 3, 4, 5, 6, 7],   # Shoulders, elbows, wrists
                'torso': [1, 8, 9, 12],       # Neck, mid-hip, hips
                'legs': [10, 11, 13, 14]      # Knees, ankles
            }
            
            # Calculate movement metrics for each body part
            for part_name, indices in body_parts.items():
                part_displacements = displacements[:, indices]
                
                # Mean movement
                mean_movement = np.nanmean(part_displacements) if not np.isnan(part_displacements).all() else 0
                movement_features.append(mean_movement)
                
                # Max movement
                max_movement = np.nanmax(part_displacements) if not np.isnan(part_displacements).all() else 0
                movement_features.append(max_movement)
                
                # Movement variation
                std_movement = np.nanstd(part_displacements) if not np.isnan(part_displacements).all() else 0
                movement_features.append(std_movement)
            
            # Calculate relative movement distribution
            total_movement = np.nansum(movement_features[::3])  # Sum of mean movements
            if total_movement > 0:
                for i in range(0, len(body_parts)):
                    relative_movement = movement_features[i * 3] / total_movement if total_movement > 0 else 0
                    movement_features.append(relative_movement)
            else:
                movement_features.extend([0] * len(body_parts))
        
        except Exception as e:
            print(f"Error calculating movement distribution: {e}")
            movement_features = [0] * 16  # Default values (4 parts * 3 features + 4 relative)
        
        return movement_features
    
    def _calculate_periodicity_features(self, pose_frames):
        """
        Calculate periodicity features using autocorrelation
        
        Args:
            pose_frames: Sequence of pose keypoints
            
        Returns:
            List of periodicity features
        """
        periodicity_features = []
        
        try:
            # Select key joints for periodicity analysis
            key_joints = [4, 7, 11, 14]  # Right wrist, left wrist, right ankle, left ankle
            
            for joint in key_joints:
                # Extract x and y coordinates over time
                x_coords = pose_frames[:, joint, 0]
                y_coords = pose_frames[:, joint, 1]
                
                # Calculate autocorrelation for x and y
                if len(x_coords) > 1:
                    # Remove NaN values
                    x_valid = x_coords[~np.isnan(x_coords)]
                    y_valid = y_coords[~np.isnan(y_coords)]
                    
                    # Only calculate if we have enough valid points
                    if len(x_valid) > 10 and len(y_valid) > 10:
                        # Normalize
                        x_norm = (x_valid - np.mean(x_valid)) / (np.std(x_valid) + 1e-10)
                        y_norm = (y_valid - np.mean(y_valid)) / (np.std(y_valid) + 1e-10)
                        
                        # Calculate autocorrelation (simplified)
                        x_auto = np.correlate(x_norm, x_norm, mode='full')
                        x_auto = x_auto[len(x_auto)//2:]
                        x_auto = x_auto / x_auto[0]
                        
                        y_auto = np.correlate(y_norm, y_norm, mode='full')
                        y_auto = y_auto[len(y_auto)//2:]
                        y_auto = y_auto / y_auto[0]
                        
                        # Find peaks in autocorrelation
                        x_peaks = self._find_peaks(x_auto)
                        y_peaks = self._find_peaks(y_auto)
                        
                        # Extract periodicity metrics
                        x_periodicity = np.mean(x_peaks) if x_peaks.size > 0 else 0
                        y_periodicity = np.mean(y_peaks) if y_peaks.size > 0 else 0
                        
                        periodicity_features.append(x_periodicity)
                        periodicity_features.append(y_periodicity)
                    else:
                        periodicity_features.extend([0, 0])
                else:
                    periodicity_features.extend([0, 0])
        
        except Exception as e:
            print(f"Error calculating periodicity features: {e}")
            periodicity_features = [0] * 8  # Default values (4 joints * 2 features)
        
        return periodicity_features
    
    def _find_peaks(self, x):
        """
        Find peaks in a 1D array
        
        Args:
            x: 1D array
            
        Returns:
            Array of peak values
        """
        # Simple peak finding
        peaks = []
        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] > x[i + 1] and x[i] > 0.2:  # Peak must be > 0.2
                peaks.append(x[i])
        
        return np.array(peaks)
    
    def _calculate_bounding_box_features(self, pose_frames):
        """
        Calculate features based on the bounding box of the pose
        
        Args:
            pose_frames: Sequence of pose keypoints
            
        Returns:
            List of bounding box features
        """
        bbox_features = []
        
        try:
            # Calculate mean pose
            mean_pose = np.mean(pose_frames, axis=0)
            
            # Find valid keypoints (confidence > 0.1)
            valid_keypoints = mean_pose[mean_pose[:, 2] > 0.1, :2]
            
            if len(valid_keypoints) > 0:
                # Calculate bounding box
                min_coords = np.min(valid_keypoints, axis=0)
                max_coords = np.max(valid_keypoints, axis=0)
                
                # Box dimensions
                width = max_coords[0] - min_coords[0]
                height = max_coords[1] - min_coords[1]
                
                # Box area
                area = width * height
                
                # Box aspect ratio
                aspect_ratio = height / width if width > 0 else 0
                
                bbox_features.extend([width, height, area, aspect_ratio])
                
                # Calculate bounding box for each frame
                boxes = []
                for i in range(pose_frames.shape[0]):
                    valid_frame_keypoints = pose_frames[i, pose_frames[i, :, 2] > 0.1, :2]
                    if len(valid_frame_keypoints) > 0:
                        min_frame = np.min(valid_frame_keypoints, axis=0)
                        max_frame = np.max(valid_frame_keypoints, axis=0)
                        box = [min_frame[0], min_frame[1], max_frame[0], max_frame[1]]
                        boxes.append(box)
                
                if boxes:
                    boxes = np.array(boxes)
                    
                    # Calculate box movement statistics
                    if len(boxes) > 1:
                        # Box center movement
                        centers = np.zeros((len(boxes), 2))
                        centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x center
                        centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y center
                        
                        center_displacements = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
                        
                        mean_center_movement = np.mean(center_displacements)
                        max_center_movement = np.max(center_displacements)
                        
                        bbox_features.extend([mean_center_movement, max_center_movement])
                        
                        # Box size changes
                        box_widths = boxes[:, 2] - boxes[:, 0]
                        box_heights = boxes[:, 3] - boxes[:, 1]
                        
                        width_changes = np.abs(box_widths[1:] - box_widths[:-1])
                        height_changes = np.abs(box_heights[1:] - box_heights[:-1])
                        
                        mean_width_change = np.mean(width_changes)
                        mean_height_change = np.mean(height_changes)
                        
                        bbox_features.extend([mean_width_change, mean_height_change])
                    else:
                        bbox_features.extend([0, 0, 0, 0])
                else:
                    bbox_features.extend([0, 0, 0, 0])
            else:
                bbox_features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        
        except Exception as e:
            print(f"Error calculating bounding box features: {e}")
            bbox_features = [0] * 8  # Default values
        
        return bbox_features
    
    def _calculate_overall_movement(self, pose_frames):
        """
        Calculate overall movement amount
        
        Args:
            pose_frames: Sequence of pose keypoints
            
        Returns:
            Overall movement value
        """
        try:
            if pose_frames.shape[0] <= 1:
                return 0
            
            # Calculate displacements between consecutive frames
            total_movement = 0
            frame_count = 0
            
            for i in range(pose_frames.shape[0] - 1):
                curr_frame = pose_frames[i, :, :2]
                next_frame = pose_frames[i + 1, :, :2]
                
                curr_valid = pose_frames[i, :, 2] > 0.1
                next_valid = pose_frames[i + 1, :, 2] > 0.1
                valid_mask = curr_valid & next_valid
                
                # Calculate total displacement for this frame pair
                frame_movement = 0
                valid_count = 0
                
                for j in range(pose_frames.shape[1]):
                    if valid_mask[j]:
                        displacement = np.linalg.norm(next_frame[j] - curr_frame[j])
                        frame_movement += displacement
                        valid_count += 1
                
                if valid_count > 0:
                    total_movement += frame_movement / valid_count
                    frame_count += 1
            
            # Average movement per frame
            if frame_count > 0:
                return total_movement / frame_count
            else:
                return 0
        
        except Exception as e:
            print(f"Error calculating overall movement: {e}")
            return 0

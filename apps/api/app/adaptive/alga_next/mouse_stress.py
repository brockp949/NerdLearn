"""
MouStress Framework - Mouse Dynamics as Cognitive Proxy

Implements the "MouStress" research framework for inferring cognitive state
from mouse motion patterns, treating arm-hand motion as a mass-spring-damper system.

Key Temporal Thresholds (from research):
- 310ms Idle Threshold: Separates active processing from distraction
- 100ms Perception Threshold: Below this feels instantaneous
- 100-200ms Micro-Hesitations: Synaptic processing of new stimuli

Behavioral Patterns:
- Flow State: High straightness ratio, optimal Fitts' Law performance
- Confusion State: Meandering paths, high curvature entropy
- Frustration State: Rage clicking, erratic high-acceleration movements

References:
- Sun et al., "MouStress: Detecting Stress from Mouse Motion" (Stanford)
- CEUR-WS: "Patterns of Confusion: Using Mouse Logs to Predict Emotional State"
- PMC: "Similarities and Differences Between Eye and Mouse Dynamics"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Deque
from enum import Enum
from datetime import datetime
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS (Research-based thresholds)
# ============================================================================

# Idle detection thresholds (milliseconds)
IDLE_THRESHOLD_MS = 310  # Active processing vs distraction boundary
PERCEPTION_THRESHOLD_MS = 100  # Below this feels instantaneous
MICRO_HESITATION_LOWER_MS = 100  # Micro-hesitation lower bound
MICRO_HESITATION_UPPER_MS = 200  # Micro-hesitation upper bound
READING_THRESHOLD_MS = 200  # Cursor parked = reading if >200ms

# Velocity thresholds (pixels per second)
SACCADE_VELOCITY_THRESHOLD = 500  # Rapid directed movement
TREMOR_VELOCITY_THRESHOLD = 50  # High-frequency low-amplitude
RAGE_CLICK_VELOCITY_THRESHOLD = 800  # Erratic high-acceleration

# Straightness thresholds
FLOW_STRAIGHTNESS_RATIO = 0.85  # Euclidean/Path distance ≈ 1.0
CONFUSION_STRAIGHTNESS_RATIO = 0.4  # Meandering path

# Curvature entropy thresholds
LOW_ENTROPY_THRESHOLD = 0.3  # Smooth, predictable
HIGH_ENTROPY_THRESHOLD = 0.8  # Chaotic, confused


class LearnerState(str, Enum):
    """Inferred learner cognitive state"""
    FLOW = "flow"  # Engaged, productive
    CONFUSION = "confusion"  # Lost, uncertain
    FRUSTRATION = "frustration"  # Angry, stuck
    FATIGUE = "fatigue"  # Tired, slowing down
    DISTRACTION = "distraction"  # Mind wandering
    READING = "reading"  # Actively processing content
    UNKNOWN = "unknown"


@dataclass
class MousePoint:
    """Single mouse position with timestamp"""
    x: int
    y: int
    timestamp_ms: float
    event_type: str = "move"  # "move", "click", "scroll"


@dataclass
class TrajectorySegment:
    """A segment of mouse trajectory between pauses"""
    points: List[MousePoint]
    start_time: float
    end_time: float

    @property
    def duration_ms(self) -> float:
        return self.end_time - self.start_time

    @property
    def path_length(self) -> float:
        """Total path length traveled"""
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.points)):
            dx = self.points[i].x - self.points[i-1].x
            dy = self.points[i].y - self.points[i-1].y
            total += np.sqrt(dx**2 + dy**2)
        return total

    @property
    def euclidean_distance(self) -> float:
        """Straight-line distance from start to end"""
        if len(self.points) < 2:
            return 0.0
        dx = self.points[-1].x - self.points[0].x
        dy = self.points[-1].y - self.points[0].y
        return np.sqrt(dx**2 + dy**2)

    @property
    def straightness_ratio(self) -> float:
        """Euclidean / Path (1.0 = perfectly straight)"""
        if self.path_length == 0:
            return 0.0
        return min(1.0, self.euclidean_distance / self.path_length)


@dataclass
class KinematicStiffness:
    """
    Kinematic stiffness indicators from mass-spring-damper model

    When stress/cognitive load increases:
    - Muscle stiffness increases
    - Higher frequency, lower amplitude movements (tremors)
    - Less smooth velocity profiles
    """
    mean_stiffness: float  # Average stiffness metric
    peak_stiffness: float  # Maximum observed
    tremor_frequency: float  # Hz of high-freq movements
    tremor_amplitude: float  # Amplitude of tremors
    smoothness_index: float  # 0-1, 1 = perfectly smooth
    stress_indicator: str  # "low", "medium", "high"


@dataclass
class TrajectoryAnalysis:
    """Complete analysis of a trajectory segment"""
    # Basic metrics
    duration_ms: float
    path_length: float
    euclidean_distance: float
    straightness_ratio: float

    # Velocity metrics
    avg_velocity: float
    peak_velocity: float
    velocity_std: float
    velocity_profile_score: float  # How bell-shaped (Fitts' Law)

    # Curvature metrics
    curvature_entropy: float
    direction_changes: int
    aoi_revisits: int  # Areas of Interest revisited

    # Temporal patterns
    micro_hesitation_count: int
    sub_movement_count: int
    idle_periods: List[float]

    # Classification
    pattern_type: str  # "straight", "hesitation", "random", "directed"
    inferred_state: LearnerState


@dataclass
class MouStressResult:
    """Complete MouStress analysis result"""
    timestamp: datetime
    analysis_window_ms: float
    trajectory_analysis: TrajectoryAnalysis
    kinematic_stiffness: KinematicStiffness
    learner_state: LearnerState
    confidence: float
    cognitive_load: str  # "low", "medium", "high"
    attention_level: str  # "low", "medium", "high"
    recommendations: List[str]


class MouStressAnalyzer:
    """
    MouStress analyzer for cognitive state inference

    Processes raw mouse events and infers learner cognitive state
    using research-based heuristics and kinematic analysis.
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        analysis_window_ms: float = 5000,  # 5 second window
    ):
        self.buffer_size = buffer_size
        self.analysis_window_ms = analysis_window_ms
        self.mouse_buffer: Deque[MousePoint] = deque(maxlen=buffer_size)
        self.click_buffer: Deque[MousePoint] = deque(maxlen=100)

        # Areas of Interest tracking
        self.aoi_history: Deque[Tuple[int, int]] = deque(maxlen=50)
        self.aoi_grid_size = 100  # pixels

        # State history for trend analysis
        self.state_history: Deque[LearnerState] = deque(maxlen=20)

        logger.info(f"MouStressAnalyzer initialized with {analysis_window_ms}ms window")

    def add_event(self, x: int, y: int, timestamp_ms: float, event_type: str = "move"):
        """Add a mouse event to the buffer"""
        point = MousePoint(x=x, y=y, timestamp_ms=timestamp_ms, event_type=event_type)

        if event_type == "click":
            self.click_buffer.append(point)
        else:
            self.mouse_buffer.append(point)

        # Track AOI
        aoi = (x // self.aoi_grid_size, y // self.aoi_grid_size)
        if not self.aoi_history or self.aoi_history[-1] != aoi:
            self.aoi_history.append(aoi)

    def analyze(self) -> MouStressResult:
        """Perform full MouStress analysis on buffered data"""
        points = list(self.mouse_buffer)

        if len(points) < 5:
            return self._create_unknown_result()

        # Get recent window
        current_time = points[-1].timestamp_ms
        window_start = current_time - self.analysis_window_ms
        recent_points = [p for p in points if p.timestamp_ms >= window_start]

        if len(recent_points) < 5:
            return self._create_unknown_result()

        # Analyze trajectory
        trajectory_analysis = self._analyze_trajectory(recent_points)

        # Analyze kinematic stiffness
        kinematic_stiffness = self._analyze_kinematic_stiffness(recent_points)

        # Infer learner state
        learner_state = self._infer_learner_state(
            trajectory_analysis, kinematic_stiffness
        )

        # Calculate confidence
        confidence = self._calculate_confidence(recent_points, trajectory_analysis)

        # Generate recommendations
        recommendations = self._generate_recommendations(learner_state, trajectory_analysis)

        # Track state history
        self.state_history.append(learner_state)

        return MouStressResult(
            timestamp=datetime.now(),
            analysis_window_ms=self.analysis_window_ms,
            trajectory_analysis=trajectory_analysis,
            kinematic_stiffness=kinematic_stiffness,
            learner_state=learner_state,
            confidence=confidence,
            cognitive_load=self._cognitive_load_from_state(learner_state),
            attention_level=self._attention_from_state(learner_state),
            recommendations=recommendations,
        )

    def _analyze_trajectory(self, points: List[MousePoint]) -> TrajectoryAnalysis:
        """Analyze trajectory characteristics"""
        # Calculate velocities
        velocities = []
        for i in range(1, len(points)):
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            dt = max(1, points[i].timestamp_ms - points[i-1].timestamp_ms)
            velocity = np.sqrt(dx**2 + dy**2) / (dt / 1000)  # pixels/second
            velocities.append(velocity)

        avg_velocity = np.mean(velocities) if velocities else 0
        peak_velocity = np.max(velocities) if velocities else 0
        velocity_std = np.std(velocities) if velocities else 0

        # Calculate path metrics
        path_length = sum(
            np.sqrt((points[i].x - points[i-1].x)**2 + (points[i].y - points[i-1].y)**2)
            for i in range(1, len(points))
        )
        euclidean_distance = np.sqrt(
            (points[-1].x - points[0].x)**2 + (points[-1].y - points[0].y)**2
        )
        straightness_ratio = euclidean_distance / path_length if path_length > 0 else 0

        # Calculate curvature entropy
        angles = self._calculate_direction_changes(points)
        curvature_entropy = np.std(angles) if angles else 0

        # Count direction changes (significant turns)
        direction_changes = sum(1 for a in angles if abs(a) > np.pi / 4)

        # Velocity profile score (how bell-shaped, per Fitts' Law)
        velocity_profile_score = self._calculate_velocity_profile_score(velocities)

        # Detect micro-hesitations (pauses 100-200ms)
        micro_hesitations = self._count_micro_hesitations(points)

        # Detect sub-movements (velocity dips during movement)
        sub_movements = self._count_sub_movements(velocities)

        # Detect idle periods (>310ms)
        idle_periods = self._detect_idle_periods(points)

        # AOI revisits
        aoi_revisits = self._count_aoi_revisits()

        # Classify pattern
        pattern_type = self._classify_pattern(
            straightness_ratio, curvature_entropy, avg_velocity
        )

        # Initial state inference
        inferred_state = LearnerState.UNKNOWN

        return TrajectoryAnalysis(
            duration_ms=points[-1].timestamp_ms - points[0].timestamp_ms,
            path_length=path_length,
            euclidean_distance=euclidean_distance,
            straightness_ratio=straightness_ratio,
            avg_velocity=avg_velocity,
            peak_velocity=peak_velocity,
            velocity_std=velocity_std,
            velocity_profile_score=velocity_profile_score,
            curvature_entropy=curvature_entropy,
            direction_changes=direction_changes,
            aoi_revisits=aoi_revisits,
            micro_hesitation_count=micro_hesitations,
            sub_movement_count=sub_movements,
            idle_periods=idle_periods,
            pattern_type=pattern_type,
            inferred_state=inferred_state,
        )

    def _analyze_kinematic_stiffness(self, points: List[MousePoint]) -> KinematicStiffness:
        """
        Analyze kinematic stiffness using mass-spring-damper model

        High cognitive load → increased muscle stiffness → detectable in:
        - Higher frequency movements
        - Lower amplitude
        - Less smooth trajectories
        """
        if len(points) < 10:
            return KinematicStiffness(
                mean_stiffness=0.5,
                peak_stiffness=0.5,
                tremor_frequency=0,
                tremor_amplitude=0,
                smoothness_index=0.5,
                stress_indicator="unknown",
            )

        # Calculate accelerations (second derivative)
        accelerations = []
        for i in range(2, len(points)):
            dt1 = max(1, points[i-1].timestamp_ms - points[i-2].timestamp_ms)
            dt2 = max(1, points[i].timestamp_ms - points[i-1].timestamp_ms)

            v1 = np.sqrt((points[i-1].x - points[i-2].x)**2 + (points[i-1].y - points[i-2].y)**2) / dt1
            v2 = np.sqrt((points[i].x - points[i-1].x)**2 + (points[i].y - points[i-1].y)**2) / dt2

            accel = (v2 - v1) / (dt2 / 1000)
            accelerations.append(abs(accel))

        # Stiffness approximation (higher acceleration variance = higher stiffness)
        accel_std = np.std(accelerations) if accelerations else 0
        mean_stiffness = min(1.0, accel_std / 5000)  # Normalize
        peak_stiffness = min(1.0, max(accelerations) / 10000) if accelerations else 0

        # Tremor detection using FFT
        if len(points) >= 32:
            # Extract positions for FFT
            x_positions = [p.x for p in points[-32:]]
            y_positions = [p.y for p in points[-32:]]

            # Detrend
            x_detrended = x_positions - np.mean(x_positions)
            y_detrended = y_positions - np.mean(y_positions)

            # FFT
            x_fft = np.abs(np.fft.fft(x_detrended))
            y_fft = np.abs(np.fft.fft(y_detrended))
            combined_fft = x_fft + y_fft

            # Find dominant frequency
            sampling_rate = 32 / (points[-1].timestamp_ms - points[-32].timestamp_ms) * 1000  # Hz
            freqs = np.fft.fftfreq(32, 1/sampling_rate)

            # Tremor is typically 8-12 Hz
            tremor_range = (freqs >= 5) & (freqs <= 15)
            tremor_power = combined_fft[tremor_range]
            tremor_frequency = freqs[tremor_range][np.argmax(tremor_power)] if len(tremor_power) > 0 else 0
            tremor_amplitude = np.max(tremor_power) if len(tremor_power) > 0 else 0
        else:
            tremor_frequency = 0
            tremor_amplitude = 0

        # Smoothness index (based on jerk - third derivative)
        jerks = []
        for i in range(1, len(accelerations)):
            dt = max(1, points[i+2].timestamp_ms - points[i+1].timestamp_ms)
            jerk = abs(accelerations[i] - accelerations[i-1]) / (dt / 1000)
            jerks.append(jerk)

        mean_jerk = np.mean(jerks) if jerks else 0
        smoothness_index = max(0, 1 - min(1, mean_jerk / 50000))

        # Determine stress indicator
        if mean_stiffness > 0.7 or tremor_amplitude > 0.5:
            stress_indicator = "high"
        elif mean_stiffness > 0.4 or tremor_amplitude > 0.25:
            stress_indicator = "medium"
        else:
            stress_indicator = "low"

        return KinematicStiffness(
            mean_stiffness=mean_stiffness,
            peak_stiffness=peak_stiffness,
            tremor_frequency=tremor_frequency,
            tremor_amplitude=tremor_amplitude,
            smoothness_index=smoothness_index,
            stress_indicator=stress_indicator,
        )

    def _infer_learner_state(
        self,
        trajectory: TrajectoryAnalysis,
        stiffness: KinematicStiffness,
    ) -> LearnerState:
        """Infer learner cognitive state from analysis"""
        # Check for rage clicking (frustration)
        recent_clicks = [c for c in self.click_buffer
                        if c.timestamp_ms > self.mouse_buffer[-1].timestamp_ms - 2000]
        if len(recent_clicks) > 5:
            return LearnerState.FRUSTRATION

        # Flow state: straight paths, smooth velocity, low stress
        if (trajectory.straightness_ratio > FLOW_STRAIGHTNESS_RATIO and
            trajectory.velocity_profile_score > 0.7 and
            stiffness.stress_indicator == "low"):
            return LearnerState.FLOW

        # Confusion state: meandering, high entropy, AOI revisits
        if (trajectory.straightness_ratio < CONFUSION_STRAIGHTNESS_RATIO and
            trajectory.curvature_entropy > HIGH_ENTROPY_THRESHOLD and
            trajectory.aoi_revisits > 3):
            return LearnerState.CONFUSION

        # Frustration: high stress, erratic movements
        if (stiffness.stress_indicator == "high" and
            trajectory.velocity_std > trajectory.avg_velocity * 0.8):
            return LearnerState.FRUSTRATION

        # Distraction: many idle periods
        long_idles = [i for i in trajectory.idle_periods if i > IDLE_THRESHOLD_MS]
        if len(long_idles) > 2:
            return LearnerState.DISTRACTION

        # Fatigue: slowing velocity, increasing stiffness over time
        if (len(self.state_history) > 5 and
            trajectory.avg_velocity < 100 and
            stiffness.smoothness_index < 0.5):
            return LearnerState.FATIGUE

        # Reading: cursor parked, occasional micro-hesitations
        if (trajectory.avg_velocity < TREMOR_VELOCITY_THRESHOLD and
            trajectory.micro_hesitation_count > 2 and
            len(long_idles) <= 1):
            return LearnerState.READING

        return LearnerState.UNKNOWN

    def _calculate_direction_changes(self, points: List[MousePoint]) -> List[float]:
        """Calculate angle changes between consecutive movement vectors"""
        angles = []
        for i in range(2, len(points)):
            v1 = np.array([points[i-1].x - points[i-2].x, points[i-1].y - points[i-2].y])
            v2 = np.array([points[i].x - points[i-1].x, points[i].y - points[i-1].y])

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)

        return angles

    def _calculate_velocity_profile_score(self, velocities: List[float]) -> float:
        """
        Score how bell-shaped the velocity profile is (Fitts' Law optimal)

        Optimal movement has smooth acceleration then deceleration
        """
        if len(velocities) < 5:
            return 0.5

        # Ideal: peak in middle, smooth rise and fall
        peak_idx = np.argmax(velocities)
        relative_peak_position = peak_idx / len(velocities)

        # Score based on peak position (should be around 0.3-0.5)
        position_score = 1 - abs(relative_peak_position - 0.4) * 2

        # Score based on symmetry
        if peak_idx > 0 and peak_idx < len(velocities) - 1:
            rise = velocities[:peak_idx]
            fall = velocities[peak_idx+1:]
            if len(rise) > 0 and len(fall) > 0:
                rise_smoothness = 1 - np.std(np.diff(rise)) / (np.mean(rise) + 0.001)
                fall_smoothness = 1 - np.std(np.diff(fall)) / (np.mean(fall) + 0.001)
                symmetry_score = (rise_smoothness + fall_smoothness) / 2
            else:
                symmetry_score = 0.5
        else:
            symmetry_score = 0.3

        return max(0, min(1, (position_score + symmetry_score) / 2))

    def _count_micro_hesitations(self, points: List[MousePoint]) -> int:
        """Count pauses between 100-200ms"""
        count = 0
        for i in range(1, len(points)):
            dt = points[i].timestamp_ms - points[i-1].timestamp_ms
            if MICRO_HESITATION_LOWER_MS <= dt <= MICRO_HESITATION_UPPER_MS:
                count += 1
        return count

    def _count_sub_movements(self, velocities: List[float]) -> int:
        """Count velocity dips during movement (sub-movements indicate cognitive interference)"""
        if len(velocities) < 3:
            return 0

        count = 0
        avg = np.mean(velocities)
        for i in range(1, len(velocities) - 1):
            # Dip: velocity drops below half of neighbors
            if velocities[i] < velocities[i-1] * 0.5 and velocities[i] < velocities[i+1] * 0.5:
                count += 1
        return count

    def _detect_idle_periods(self, points: List[MousePoint]) -> List[float]:
        """Detect idle periods exceeding threshold"""
        idle_periods = []
        for i in range(1, len(points)):
            dt = points[i].timestamp_ms - points[i-1].timestamp_ms
            if dt > IDLE_THRESHOLD_MS:
                idle_periods.append(dt)
        return idle_periods

    def _count_aoi_revisits(self) -> int:
        """Count how many AOIs were revisited"""
        if len(self.aoi_history) < 3:
            return 0

        aoi_list = list(self.aoi_history)
        revisits = 0
        seen = set()
        for aoi in aoi_list:
            if aoi in seen:
                revisits += 1
            seen.add(aoi)
        return revisits

    def _classify_pattern(
        self,
        straightness: float,
        entropy: float,
        velocity: float,
    ) -> str:
        """Classify trajectory pattern type"""
        if straightness > FLOW_STRAIGHTNESS_RATIO:
            return "straight"
        elif entropy > HIGH_ENTROPY_THRESHOLD:
            return "random"
        elif velocity < TREMOR_VELOCITY_THRESHOLD:
            return "hesitation"
        else:
            return "directed"

    def _calculate_confidence(
        self,
        points: List[MousePoint],
        trajectory: TrajectoryAnalysis,
    ) -> float:
        """Calculate confidence in state inference"""
        # More data = higher confidence
        data_confidence = min(1.0, len(points) / 50)

        # Clear patterns = higher confidence
        pattern_confidence = 0.5
        if trajectory.pattern_type == "straight":
            pattern_confidence = 0.9 if trajectory.straightness_ratio > 0.9 else 0.7
        elif trajectory.pattern_type == "random":
            pattern_confidence = 0.8 if trajectory.curvature_entropy > 1.0 else 0.6

        return (data_confidence + pattern_confidence) / 2

    def _cognitive_load_from_state(self, state: LearnerState) -> str:
        """Map learner state to cognitive load level"""
        mapping = {
            LearnerState.FLOW: "medium",
            LearnerState.CONFUSION: "high",
            LearnerState.FRUSTRATION: "high",
            LearnerState.FATIGUE: "high",
            LearnerState.DISTRACTION: "low",
            LearnerState.READING: "medium",
            LearnerState.UNKNOWN: "medium",
        }
        return mapping.get(state, "medium")

    def _attention_from_state(self, state: LearnerState) -> str:
        """Map learner state to attention level"""
        mapping = {
            LearnerState.FLOW: "high",
            LearnerState.CONFUSION: "medium",
            LearnerState.FRUSTRATION: "low",
            LearnerState.FATIGUE: "low",
            LearnerState.DISTRACTION: "low",
            LearnerState.READING: "high",
            LearnerState.UNKNOWN: "medium",
        }
        return mapping.get(state, "medium")

    def _generate_recommendations(
        self,
        state: LearnerState,
        trajectory: TrajectoryAnalysis,
    ) -> List[str]:
        """Generate pedagogical recommendations based on state"""
        recommendations = []

        if state == LearnerState.CONFUSION:
            recommendations.append("Consider switching to a simpler modality (e.g., video)")
            recommendations.append("Offer scaffolding or hints")
            if trajectory.aoi_revisits > 3:
                recommendations.append("User may be looking for specific information - offer search")

        elif state == LearnerState.FRUSTRATION:
            recommendations.append("Take a break prompt recommended")
            recommendations.append("Switch to passive content (video/podcast)")
            recommendations.append("Reduce difficulty level")

        elif state == LearnerState.FATIGUE:
            recommendations.append("Suggest ending session or taking break")
            recommendations.append("Switch to lighter content")
            recommendations.append("Use shorter, chunked content")

        elif state == LearnerState.DISTRACTION:
            recommendations.append("Attention prompt may help")
            recommendations.append("Consider more engaging modality")
            recommendations.append("Gamification elements may help focus")

        elif state == LearnerState.FLOW:
            recommendations.append("Maintain current difficulty level")
            recommendations.append("User is engaged - minimal intervention")

        elif state == LearnerState.READING:
            recommendations.append("User is actively processing - do not interrupt")

        return recommendations

    def _create_unknown_result(self) -> MouStressResult:
        """Create result when insufficient data"""
        return MouStressResult(
            timestamp=datetime.now(),
            analysis_window_ms=self.analysis_window_ms,
            trajectory_analysis=TrajectoryAnalysis(
                duration_ms=0, path_length=0, euclidean_distance=0,
                straightness_ratio=0, avg_velocity=0, peak_velocity=0,
                velocity_std=0, velocity_profile_score=0, curvature_entropy=0,
                direction_changes=0, aoi_revisits=0, micro_hesitation_count=0,
                sub_movement_count=0, idle_periods=[], pattern_type="unknown",
                inferred_state=LearnerState.UNKNOWN,
            ),
            kinematic_stiffness=KinematicStiffness(
                mean_stiffness=0.5, peak_stiffness=0.5, tremor_frequency=0,
                tremor_amplitude=0, smoothness_index=0.5, stress_indicator="unknown",
            ),
            learner_state=LearnerState.UNKNOWN,
            confidence=0.0,
            cognitive_load="unknown",
            attention_level="unknown",
            recommendations=["Insufficient data for analysis"],
        )

    def get_state_trend(self) -> Dict[str, int]:
        """Get distribution of recent states"""
        counts = {}
        for state in self.state_history:
            counts[state.value] = counts.get(state.value, 0) + 1
        return counts

    def clear(self):
        """Clear all buffers"""
        self.mouse_buffer.clear()
        self.click_buffer.clear()
        self.aoi_history.clear()
        self.state_history.clear()

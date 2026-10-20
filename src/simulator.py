"""
SentinelAI - Sensor Stream Simulator Module.
Demonstrates dynamic telemetry processing by replaying historical NASA C-MAPSS engine
trajectories or generating controllable synthetic sensor signals without claiming live factory connectivity.

STRICT DISCLAIMER:
This module is a demonstration simulation harness. It simulates stream ingestion
within the application runtime. Production industrial architectures deploy dedicated
distributed stream brokers (e.g., Apache Kafka, MQTT, AWS Kinesis) and stateful stream
processors (e.g., Apache Flink).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import SENSOR_COLUMNS, HEALTH_THRESHOLDS
from src.health_analysis import assess_observation_health


SIMULATOR_DISCLAIMER = (
    "DEMO SIMULATION MODE — FOR DEMONSTRATION PURPOSES ONLY. "
    "Simulates dynamic observation intake using replayed historical C-MAPSS runs or synthetic "
    "perturbations. This is not a live physical connection to industrial machinery or factory PLCs."
)


@dataclass
class StreamObservation:
    """Represents a single incoming telemetry cycle record."""
    cycle: int
    engine_id: int
    sensor_values: Dict[str, float]
    source_type: str  # 'replayed_historical' or 'synthetic_demonstration'
    is_injected_anomaly: bool = False


@dataclass
class SimulationResult:
    """Result of running an incoming observation through the frozen anomaly pipeline."""
    cycle: int
    engine_id: int
    sensor_values: Dict[str, float]
    source_type: str
    is_injected_anomaly: bool
    is_anomaly_if: int
    anomaly_score: float
    is_anomaly_base: int
    baseline_score: float
    is_anomaly: int
    triggering_sensors: str
    health_status: str
    consecutive_anomalies: int
    action_recommendation: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S.%f")[:-3])


class SensorStreamSimulator:
    """
    Stateful streaming simulator that replays historical engine cycles or generates
    synthetic sensor signals, passing each observation through frozen trained models.
    """

    def __init__(
        self,
        mode: str = "HISTORICAL_REPLAY",
        engine_id: int = 1,
        historical_df: Optional[pd.DataFrame] = None,
        noise_level: float = 0.02,
        anomaly_injection_cycle: int = 35,
        target_sensor: str = "sensor_2",
        anomaly_shift: float = 4.0,
    ):
        """
        Initialize the stream simulator.

        Parameters
        ----------
        mode : str
            'HISTORICAL_REPLAY' to replay real historical cycles in original order,
            or 'SYNTHETIC_INJECTION' to generate synthetic cycles with injected drift.
        engine_id : int
            Engine unit identifier to replay or tag.
        historical_df : pd.DataFrame, optional
            Telemetry dataframe containing historical records for the selected engine.
        noise_level : float
            Standard deviation of Gaussian noise added in synthetic mode.
        anomaly_injection_cycle : int
            Cycle threshold at which anomalous drift/spikes are injected in synthetic mode.
        target_sensor : str
            Sensor channel to perturb in synthetic injection mode.
        anomaly_shift : float
            Magnitude of deviation injected in synthetic mode.
        """
        self.mode = mode.upper()
        self.engine_id = int(engine_id)
        self.noise_level = float(noise_level)
        self.anomaly_injection_cycle = int(anomaly_injection_cycle)
        self.target_sensor = target_sensor
        self.anomaly_shift = float(anomaly_shift)

        self.current_step: int = 0
        self.is_running: bool = False
        self.is_paused: bool = False
        self.history: List[SimulationResult] = []
        self._consecutive_streak: int = 0

        # Prepare historical data slice if provided
        self._records: List[Dict[str, Any]] = []
        if self.mode == "HISTORICAL_REPLAY" and historical_df is not None:
            engine_records = historical_df[historical_df["engine_id"] == self.engine_id].sort_values("cycle")
            self._records = engine_records.to_dict(orient="records")

    @property
    def total_available_cycles(self) -> int:
        """Returns the total number of available cycles in historical mode, or default for synthetic."""
        if self.mode == "HISTORICAL_REPLAY":
            return len(self._records)
        return 100  # Default cap for synthetic stream demonstration

    @property
    def is_finished(self) -> bool:
        """Returns True if the historical stream has reached its final cycle."""
        return self.current_step >= self.total_available_cycles

    def reset(self) -> None:
        """Resets the simulator state back to cycle 0 and clears the event log."""
        self.current_step = 0
        self.is_running = False
        self.is_paused = False
        self.history.clear()
        self._consecutive_streak = 0

    def start(self) -> None:
        """Transitions simulator state to actively running."""
        self.is_running = True
        self.is_paused = False

    def pause(self) -> None:
        """Pauses the stream without resetting processed history."""
        self.is_running = False
        self.is_paused = True

    def step(self, preprocessor, if_model, base_model) -> Optional[SimulationResult]:
        """
        Processes a single incoming observation through the frozen anomaly detection pipeline.
        Strictly preserves model weights without retraining.

        Returns
        -------
        SimulationResult or None if the stream has finished.
        """
        if self.is_finished:
            self.is_running = False
            return None

        # 1. Fetch or generate the incoming observation
        if self.mode == "HISTORICAL_REPLAY":
            row_dict = self._records[self.current_step].copy()
            cycle = int(row_dict["cycle"])
            source_type = "replayed_historical"
            is_injected = False
            sensor_vals = {col: float(row_dict[col]) for col in SENSOR_COLUMNS if col in row_dict}
            row_df = pd.DataFrame([row_dict])

        else:  # SYNTHETIC_INJECTION
            cycle = self.current_step + 1
            source_type = "synthetic_demonstration"
            is_injected = cycle >= self.anomaly_injection_cycle

            # Generate synthetic nominal sensor values with slight operational noise
            rng = np.random.default_rng(seed=cycle + 1000)
            nominal_base = {
                "sensor_2": 642.5, "sensor_3": 1585.0, "sensor_4": 1400.0,
                "sensor_6": 21.6, "sensor_7": 553.5, "sensor_8": 2388.05,
                "sensor_9": 9050.0, "sensor_11": 47.3, "sensor_12": 521.5,
                "sensor_13": 2388.0, "sensor_14": 8130.0, "sensor_15": 8.42,
                "sensor_17": 392.0, "sensor_20": 38.8, "sensor_21": 23.3,
            }
            sensor_vals = {}
            row_dict = {"engine_id": self.engine_id, "cycle": cycle, "op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0}
            
            for s_name, base_v in nominal_base.items():
                noise = rng.normal(0, base_v * self.noise_level * 0.01)
                val = base_v + noise
                if is_injected and s_name == self.target_sensor:
                    # Inject controlled thermal or pressure drift
                    val += base_v * (self.anomaly_shift * 0.01) * ((cycle - self.anomaly_injection_cycle + 1) ** 0.5)
                sensor_vals[s_name] = round(val, 3)
                row_dict[s_name] = sensor_vals[s_name]

            # Populate constant sensors for schema completeness
            for s in [1, 5, 10, 16, 18, 19]:
                row_dict[f"sensor_{s}"] = 0.0

            row_df = pd.DataFrame([row_dict])

        # 2. Run observation through frozen pipeline (NO RETRAINING)
        x_scaled, _ = preprocessor.transform(row_df)
        if_preds, if_scores = if_model.predict(x_scaled)
        base_preds, base_scores, base_audit = base_model.predict(row_df)

        is_if = int(if_preds[0])
        score_if = float(if_scores[0])
        is_base = int(base_preds[0])
        score_base = float(base_scores[0])
        is_anom = 1 if (is_if == 1 or is_base == 1) else 0

        if isinstance(base_audit, pd.DataFrame) and not base_audit.empty and "triggering_sensors" in base_audit.columns:
            triggers = str(base_audit["triggering_sensors"].iloc[0])
        elif isinstance(base_audit, (list, tuple)) and len(base_audit) > 0:
            triggers = str(base_audit[0])
        else:
            triggers = "None"

        # Update streak
        if is_anom == 1:
            self._consecutive_streak += 1
        else:
            self._consecutive_streak = 0

        trigger_count = int(base_audit["trigger_count"].iloc[0]) if (isinstance(base_audit, pd.DataFrame) and not base_audit.empty and "trigger_count" in base_audit.columns) else (0 if triggers == "None" else len(triggers.split(";")))

        # Assess health status using documented rules
        health_status, _, action = assess_observation_health(
            anomaly_score=score_if,
            baseline_triggers=trigger_count,
            consecutive_high_scores=self._consecutive_streak,
        )

        res = SimulationResult(
            cycle=cycle,
            engine_id=self.engine_id,
            sensor_values=sensor_vals,
            source_type=source_type,
            is_injected_anomaly=is_injected,
            is_anomaly_if=is_if,
            anomaly_score=score_if,
            is_anomaly_base=is_base,
            baseline_score=score_base,
            is_anomaly=is_anom,
            triggering_sensors=triggers,
            health_status=health_status,
            consecutive_anomalies=self._consecutive_streak,
            action_recommendation=action,
        )

        self.history.append(res)
        self.current_step += 1
        return res

    def get_history_df(self) -> pd.DataFrame:
        """Returns all processed simulated observations as a structured DataFrame."""
        if not self.history:
            return pd.DataFrame()
        
        rows = []
        for r in self.history:
            item = {
                "timestamp": r.timestamp,
                "engine_id": r.engine_id,
                "cycle": r.cycle,
                "source_type": r.source_type,
                "is_injected_anomaly": r.is_injected_anomaly,
                "health_status": r.health_status,
                "anomaly_score": r.anomaly_score,
                "baseline_score": r.baseline_score,
                "is_anomaly": r.is_anomaly,
                "is_anomaly_if": r.is_anomaly_if,
                "is_anomaly_base": r.is_anomaly_base,
                "consecutive_anomalies": r.consecutive_anomalies,
                "triggering_sensors": r.triggering_sensors,
                "action_recommendation": r.action_recommendation,
            }
            # Add sensor values
            item.update(r.sensor_values)
            rows.append(item)
        return pd.DataFrame(rows)

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Returns live summary counts of the simulated session."""
        if not self.history:
            return {
                "cycles_processed": 0,
                "total_anomalies": 0,
                "anomaly_rate": 0.0,
                "current_health": "HEALTHY",
                "peak_score": 0.0,
                "critical_events": 0,
            }

        total_obs = len(self.history)
        anom_count = sum(r.is_anomaly for r in self.history)
        rate = (anom_count / total_obs) * 100 if total_obs > 0 else 0.0
        peak = max(r.anomaly_score for r in self.history)
        crit_count = sum(1 for r in self.history if r.health_status == "CRITICAL")
        latest_health = self.history[-1].health_status

        return {
            "cycles_processed": total_obs,
            "total_anomalies": anom_count,
            "anomaly_rate": rate,
            "current_health": latest_health,
            "peak_score": peak,
            "critical_events": crit_count,
        }

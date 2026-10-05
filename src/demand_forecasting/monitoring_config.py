"""Shared settings for collector, inference and hourly Airflow validation."""
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class MonitorConfig:
    interval_hours: int = 1
    delay_hours: int = 1
    recheck_hours: int = 48
    label_wait_seconds: int = 120
    issue_grace_seconds: int = 120
    feature_delay_warning_seconds: int = 900
    feature_missing_warning_ratio: float = .05
    serving_window_hours: int = 24
    min_samples: int = 100
    min_demand: float = 1000.0
    psi_warning: float = .2
    range_warning_ratio: float = .05
    max_rows: int = 1000000
    poll_seconds: int = 60

    @classmethod
    def from_env(cls):
        values={}
        for name,field in cls.__dataclass_fields__.items():
            default=field.default
            values[name]=type(default)(os.getenv('MONITOR_'+name.upper(),str(default)))
        obj=cls(**values)
        if obj.interval_hours not in (1,2): raise ValueError('MONITOR_INTERVAL_HOURS must be 1 or 2')
        if min(obj.delay_hours,obj.serving_window_hours,obj.min_samples,obj.max_rows)<1:
            raise ValueError('Delay/window/sample/row settings must be positive')
        if obj.recheck_hours<obj.interval_hours or obj.recheck_hours>168:
            raise ValueError('MONITOR_RECHECK_HOURS must be interval_hours..168')
        if min(obj.label_wait_seconds,obj.issue_grace_seconds,obj.feature_delay_warning_seconds,obj.min_demand,obj.psi_warning)<0:
            raise ValueError('Monitoring thresholds cannot be negative')
        if not 0<=obj.feature_missing_warning_ratio<=1 or not 0<=obj.range_warning_ratio<=1:
            raise ValueError('Ratios must be in [0,1]')
        if obj.poll_seconds<10: raise ValueError('MONITOR_POLL_SECONDS must be >=10')
        return obj

"""At 05:00 evaluate forecast origins [03:00,04:00), including H60 labels."""
from datetime import timedelta
import os
import pendulum
from airflow.sdk import dag,task
from demand_forecasting.monitoring_config import MonitorConfig

config=MonitorConfig.from_env()

@dag(dag_id='demand_model_monitoring',schedule=f'0 */{config.interval_hours} * * *',
     start_date=pendulum.datetime(2026,1,1,tz=os.getenv('LOCAL_TIMEZONE','Asia/Ho_Chi_Minh')),
     catchup=False,max_active_runs=1,is_paused_upon_creation=False,
     default_args={'owner':'demand-forecasting','retries':1,'retry_delay':timedelta(minutes=5)},
     tags=['demand','monitoring'])
def demand_model_monitoring():
    @task(execution_timeout=timedelta(minutes=50))
    def validate_delayed_windows():
        from demand_forecasting.performance_monitor import run
        from demand_forecasting.monitoring_store import monitored_job
        return monitored_job('performance')(run)()
    validate_delayed_windows()

demand_model_monitoring()

import schedule
import time
import pendulum
from main import read_data

def job():
    df = pd.DataFrame()  # Replace with actual data reading logic
    ts = pendulum.now().to_iso8601_string()
    monitor_data(df, ts)

# Schedule the job to run every 10 minutes
schedule.every(10).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
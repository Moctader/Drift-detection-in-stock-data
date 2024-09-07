# import schedule
# import time
# import pendulum
# from main import read_data

# def job():
#     df = pd.DataFrame()  # Replace with actual data reading logic
#     ts = pendulum.now().to_iso8601_string()
#     monitor_data(df, ts)

# # Schedule the job to run every 10 minutes
# schedule.every(10).minutes.do(job)

# while True:
#     schedule.run_pending()
#     time.sleep(1)

import os
from prefect import flow, get_client

# Set the Prefect API URL
os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

@flow
async def test_connection():
    try:
        client = get_client()
        health = await client.api_healthcheck()
        print(f"Health check response: {health}")
        if health and health.status == "ok":
            print("Prefect client is configured correctly and can reach the server!")
        else:
            print("Prefect client is configured, but the server health check failed or returned an unexpected response.")
    except Exception as e:
        print(f"Failed to connect to the Prefect server: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_connection())
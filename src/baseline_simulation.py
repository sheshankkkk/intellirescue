import random
import statistics


def run_baseline_simulation(
    hours: int = 24,
    incidents_per_hour: int = 3,
    avg_response_time_min: float = 18.0,
):
    """
    Very simple simulation:
    - Every hour, a fixed number of incidents appear.
    - Response time is sampled around avg_response_time_min.
    - At the end, we print average response time and total incidents.

    In later phases, this will be replaced by:
    - realistic incident generation
    - graph-based travel times
    - RL dispatch policy
    """
    random.seed(42)
    response_times = []
    total_incidents = 0

    for hour in range(hours):
        for _ in range(incidents_per_hour):
            total_incidents += 1
            rt = random.gauss(mu=avg_response_time_min, sigma=5.0)
            rt = max(1.0, rt)  # avoid negative or zero
            response_times.append(rt)

    avg_rt = statistics.mean(response_times)
    p95_rt = statistics.quantiles(response_times, n=20)[-1]  # ~95th percentile

    print("=== IntelliRescue Baseline Simulation (Phase 1) ===")
    print(f"Total incidents       : {total_incidents}")
    print(f"Average response time : {avg_rt:.2f} minutes")
    print(f"95th percentile       : {p95_rt:.2f} minutes")


if __name__ == "__main__":
    run_baseline_simulation()

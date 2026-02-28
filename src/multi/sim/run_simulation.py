"""Run SimulationAgent and print the report (history + summary) to stdout."""
from src.multi.sim import SimulationAgent

if __name__ == "__main__":
    agent = SimulationAgent(max_iterations=5)
    _, output = agent.run_and_report(initial_cooling_rate_K_per_min=15.0)
    print(output)

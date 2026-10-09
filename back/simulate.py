# Delete all existing recordings and regenerate every example world.
import shutil
from pathlib import Path
from example_worlds import simulate_three_body, simulate_billard_ball


def main():
    logs_dir = Path(__file__).parent.parent / 'logs'
    if logs_dir.exists():
        print(f'Wiping {logs_dir}')
        shutil.rmtree(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print('=== Three Body ===')
    simulate_three_body()

    print('=== Billard Ball ===')
    simulate_billard_ball()

    print('=== Done ===')


if __name__ == '__main__':
    main()

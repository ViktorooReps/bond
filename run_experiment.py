import json
import subprocess
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawTextHelpFormatter
from datetime import datetime
from time import sleep
from typing import List


class DefaultHelpRawFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    pass


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Run experiment from configuration.\n\nRun configuration example (contents of file config.json):\n'
                                        '[\n'
                                        '  {\n'
                                        '    "script": "some_script.py",\n'
                                        '    "args": [\n'
                                        '      "arg",\n'
                                        '      "--optional=123"\n'
                                        '    ]\n'
                                        '  },\n'
                                        '  {\n'
                                        '    "script": "some_other_script.py",\n'
                                        '    "args": [\n'
                                        '      "arg",\n'
                                        '      "--optional=321"\n'
                                        '    ]\n'
                                        '  }\n'
                                        ']\n\n'
                                        'script: path to python script\n'
                                        'args: (optional) arguments for python script\n', formatter_class=DefaultHelpRawFormatter)

    # positional
    parser.add_argument('config', metavar='JSON RUN CONFIGURATION', type=str, nargs='+',
                        help='Path to file with run configuration.')

    # optional
    parser.add_argument('--sleep', type=int, default=10,
                        help='Sleep time in seconds between runs.')

    return parser


if __name__ == '__main__':
    arg_parser = create_parser()

    args = arg_parser.parse_args()

    for config_file_name in args.config:

        with open(config_file_name) as config_file:
            config: List[dict] = json.load(config_file)

        print(f'Running {len(config)} scripts from {config_file_name}...')

        for config_idx, run_config in enumerate(config):
            script_path = run_config['script']
            script_args: List[str] = ['python', script_path] + list(map(str, run_config.get('args', [])))

            print(f'\nRunning config №{config_idx + 1}...')
            print(f'\tscript name: {script_path}')
            if 'args' in run_config:
                print(f'\tscript arguments: {run_config["args"]}')

            print('-' * 40 + f'START\t{datetime.now()}')
            subprocess.run(script_args)
            sleep(args.sleep)
            print('-' * 40 + f'END\t{datetime.now()}')

        print('\n\n')

import io
import json
import os
import subprocess
import sys
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

            if 'save_output' not in run_config:
                print('-' * 40 + f'START\t{datetime.now()}')
                subprocess.run(script_args)
                print('-' * 40 + f'END\t{datetime.now()}')
                print('Argument save_output is not specified! Output will be lost!')
            else:
                print('-' * 40 + f'START\t{datetime.now()}')

                log_filename = run_config['save_output']
                os.makedirs(os.path.dirname(log_filename), exist_ok=True)
                with io.open(log_filename, 'wb') as log_file, io.open(log_filename, 'rb') as reader:
                    process = subprocess.Popen(script_args, stdout=log_file)
                    while process.poll() is None:
                        sys.stdout.write(reader.read().decode(sys.stdout.encoding))
                        sleep(0.1)
                    sys.stdout.write(reader.read().decode(sys.stdout.encoding))

                print('-' * 40 + f'END\t{datetime.now()}')
                print(f'Saved script output to {log_filename}.')

            if args.sleep > 0:
                print(f'Sleeping for {args.sleep} seconds...')
                sleep(args.sleep)

        print('\n\n')

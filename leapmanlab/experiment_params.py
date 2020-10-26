"""General experiment argument parser that allows parsing of command-line args
or a config file.

"""
import ast
import sys

from typing import Any, Dict, Optional, Sequence


def experiment_params(default_params: Optional[Dict[str, Any]]=None,
                      required_keys: Optional[Sequence[str]]=None) \
        -> Dict[str, Any]:
    """Builds a dictionary of experiment parameters from user input (
    commandline or config file) and default values.
    
    Generates a dictionary of experiment parameters from either (1) a sequence
    of command-line arguments in Pythonic key=value format, or (2) a single
    command-line argument specifying a path to a config file. The dictionary
    can be supplemented by optional default values passed as the
    `default_params` argument. To require that certain keys be passed to the
    experiment, a list of keys can optionally be passed as the
    `required_keys` argument.
    
    Args:
        default_params (Optional[Dict[str, Any]]): A dictionary of default
            experiment param keys and values that are inserted into the parsed
            params dict if not supplied by the user. If default_params[key]
            is itself a Dict for some key, the default and user-supplied
            params are merged, with user-supplied values overwriting default
            ones, i.e. {**default_dict, **user_dict}.
        required_keys (Optional[Sequence[str]]): A list of required
            experiment param keys. If any keys are supplied to this arg,
            this function will raise a KeyError if the params dict does not
            contain those keys.

    Returns:
        (Dict[str, Any]): Dictionary of experiment parameters parsed from args
            or a config file.
    
    """
    args = sys.argv
    if len(args) == 2 and '=' not in args[1]:
        # Only one argument and it's not in key=value form. Assume it is a
        # path to a config file
        params = config_parser(args[1])
    else:
        # Assume we have key=value format args
        params = kwarg_parser(args)

    # Add default params to the params dict
    if default_params:
        for key in default_params:
            # Default value associated with the key
            default_value = default_params[key]
            if isinstance(default_value, Dict):
                # If the value is a dict, merge with the user-supplied dict
                # (if any)
                if key in params:
                    params[key] = {**default_value,
                                   **params[key]}
                else:
                    params[key] = {**default_value}
            else:
                # Insert the default value into the params dict if
                # not already present
                if key not in params:
                    params[key] = default_value

    # Check that the required param keys are present in the params
    # dict
    if required_keys:
        for key in required_keys:
            if key not in params:
                raise KeyError(f'Required key {key} not found in '
                               f'params')

    return params


def config_parser(fname: str) -> Dict[str, Any]:
    """

    Args:
        fname (str): Name of the config file

    Returns:
        (Dict[str, Any]): Dictionary of parsed config args.

    """
    parsed_kwargs = {}
    with open(fname, 'r') as file:
        lines = [line for line in file.read().split('\n') if len(line) > 0]

    for line in lines:
        if '=' in line:
            split = line.split('=')
            parsed_kwargs[split[0]] = ast.literal_eval(
                split[1].replace('\n', ''))
        else:
            print(f'Warning: Ignoring config line {line}')

    return parsed_kwargs


def kwarg_parser(args):
    """Parse python-style keyword args from the command-line.
    Args:
        args: CLI arguments.
    Returns:
        parsed_kwargs: Dict of key-value pairs.
    """
    parsed_kwargs = {}
    for arg in args[1:]:
        if '=' in arg:
            split = arg.split('=')
            parsed_kwargs[split[0]] = ast.literal_eval(split[1])
    return parsed_kwargs
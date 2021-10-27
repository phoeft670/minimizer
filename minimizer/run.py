import logging
import os
import resource
import subprocess
import sys
import errno

from lab import tools
from lab.calls.call import set_limit


class Run:
    """Define an executable command with time and memory limits.
    """

    def __init__(self, command, time_limit=1800, memory_limit=None, log_output=None):
        """*command* is a list of strings that starts your program with
        the desired parameters on a Linux machine.

        After *time_limit* seconds, the subprocess of *command* 
        is killed.

        Above a memory usage of *memory_limit* MiB, the subprocess of
        *command* is killed.

        Use the *log_output* option ``"on_fail"`` if you want log files to be
        written when *command* terminates on a non-zero exit code or use the
        option ``"always"`` if you want them always to be written. These options
        only work in combination with the :func:`minimizer.run.run_all` function.
        """
        self.command = command
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.log_on_fail = True if log_output == "on_fail" else False
        self.log_always = True if log_output == "always" else False

    def __repr__(self):
        return f'Run(\"{" ".join([os.path.basename(part) for part in self.command])}\")'

    def start(self, state):
        """Format the command with the entries of *state* and execute it with
        `subprocess.Popen <https://docs.python.org/3/library/subprocess.html#subprocess.Popen>`_.
        Return the 3-tuple (stdout, stderr, returncode) with the values obtained 
        from the executed command.
        """
        # These declarations are needed for the _prepare_call() function.
        time_limit = self.time_limit
        memory_limit = self.memory_limit

        def _prepare_call():
            # When the soft time limit is reached, SIGXCPU is emitted. Once we
            # reach the higher hard time limit, SIGKILL is sent. Having some
            # padding between the two limits allows programs to handle SIGXCPU.
            if time_limit is not None:
                set_limit(resource.RLIMIT_CPU, time_limit, time_limit + 5)
            if memory_limit is not None:
                _, hard_mem_limit = resource.getrlimit(resource.RLIMIT_AS)
                # Convert memory from MiB to Bytes.
                set_limit(resource.RLIMIT_AS, memory_limit *
                          1024 * 1024, hard_mem_limit)
            set_limit(resource.RLIMIT_CORE, 0, 0)

        formatted_command = [part.format(**state) for part in self.command]
        logging.debug(f"Formatted command:\n{formatted_command}")

        cwd = state["cwd"] if "cwd" in state else None

        try:
            process = subprocess.Popen(formatted_command,
                                       preexec_fn=_prepare_call,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       text=True,
                                       cwd=cwd)
        except OSError as err:
            if err.errno == errno.ENOENT:
                sys.exit('Error: Call "{}" failed. One of the files was not found.'.format(
                    ' '.join(formatted_command)))
            else:
                raise

        out_str, err_str = process.communicate()

        return (out_str, err_str, process.returncode)


class RunWithInputFile(Run):
    """Extension of the :class:`Run <minimizer.run.Run>` class adding
    the option of sending the content of a file to stdin.
    """
    # e.g., in a command like ``path/to/./my_executable < my_input_file``.

    def __init__(self, command, input_file, **kwargs):
        """*input_file* is the path to the file whose content should be sent to
        the stdin of the executed *command*.
        """
        super().__init__(command, **kwargs)
        self.input_file = input_file

    def start(self, state):
        """Same as the :meth:`base method <minimizer.run.Run.start>`, with
        the addition of the content from *input_file* being passed to the
        stdin of the executed *command*.
        """
        # These declarations are needed for the _prepare_call() function.
        time_limit = self.time_limit
        memory_limit = self.memory_limit

        def _prepare_call():
            # When the soft time limit is reached, SIGXCPU is emitted. Once we
            # reach the higher hard time limit, SIGKILL is sent. Having some
            # padding between the two limits allows programs to handle SIGXCPU.
            if time_limit is not None:
                set_limit(resource.RLIMIT_CPU, time_limit, time_limit + 5)
            if memory_limit is not None:
                _, hard_mem_limit = resource.getrlimit(resource.RLIMIT_AS)
                # Convert memory from MiB to Bytes.
                set_limit(resource.RLIMIT_AS, memory_limit *
                          1024 * 1024, hard_mem_limit)
            set_limit(resource.RLIMIT_CORE, 0, 0)

        formatted_command = [part.format(**state) for part in self.command]

        cwd = state["cwd"] if "cwd" in state else None

        try:
            process = subprocess.Popen(formatted_command,
                                       preexec_fn=_prepare_call,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       stdin=subprocess.PIPE,
                                       text=True,
                                       cwd=cwd)
        except OSError as err:
            if err.errno == errno.ENOENT:
                sys.exit('Error: Call "{}" failed. One of the files was not found.'.format(
                    ' '.join(formatted_command)))
            else:
                raise

        f = open(self.input_file.format(**state), "r")
        input_text = f.read()
        f.close()

        out_str, err_str = process.communicate(input=input_text)

        return (out_str, err_str, process.returncode)


def __run_run(state, results, name, run):
    stdout, stderr, returncode = run.start(state)
    if run.log_always or run.log_on_fail and returncode != 0:
        cwd = state["cwd"] if "cwd" in state else os.path.dirname(
                tools.get_script_path())
        if stdout:
            with open(os.path.join(cwd, f"{name}.log"), "w") as logfile:
                logfile.write(stdout)
        if stderr:
            with open(os.path.join(cwd, f"{name}.err"), "w") as errfile:
                errfile.write(stderr)
    results.update(
            {name: {"stdout": stdout, "stderr": stderr, "returncode": returncode}}
        )
"""
def __parse_run(parsers, parsed_results, name, result):
    parsed_results.update(
            {name: {"stdout": {}, "stderr": {},
                    "returncode": result["returncode"]}}
        )
    for parser in parsers:
        parsed_results[name]["stdout"].update(
                parser.parse(name, result["stdout"]))
        parsed_results[name]["stderr"].update(
                parser.parse(name, result["stderr"]))
"""

def __parse_run(parsers, parsed_results, name, result):
    for parser in parsers:
        parsed_results[name].update(
                parser.parse(name, result["stdout"]))
        parsed_results[name].update(
                parser.parse(name, result["stderr"]))

def run_all(state):
    """Start all runs in *state["runs"]* and return a *results* dictionary
    where run outputs of run *run_name* can be accessed via:
    
    - *results[run_name]["stdout"]*,
    - *results[run_name]["stderr"]* and
    - *results[run_name]["returncode"]*.
    """
    assert "runs" in state, "Could not find entry \"runs\" in state."
    results = {}
    for name, run in state["runs"].items():
        __run_run(state, results, name, run)
    return results


def run_and_parse_all(state, parsers):
    """Execute :func:`minimizer.run.run_all` and apply all *parsers* to the
    generated stdout and stderr outputs. Return an updated version of the
    *results* dictionary containing the parsing results in place of the actual
    stdout and stderr outputs. *parsers* can be a list of :class:`minimizer.parser.Parser`
    objects or a single one.
    """
    results = run_all(state)
    parsed_results = {}
    parsers = [parsers] if not isinstance(parsers, list) else parsers
    for name, result in results.items():
        __parse_run(parsers, parsed_results, name, result)
    parsed_results["raw_results"] = results
    return parsed_results

def run_and_parse_sequentially(state, parsers, order):
    parsers = [parsers] if not isinstance(parsers, list) else parsers
    for name in order:
        __run_run(state,state,name,state["runs"][name])
        __parse_run(parsers,state,name,state[name])
    return state
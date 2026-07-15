from functools import wraps
from json import JSONDecodeError
from typing import Any, Callable
from argparse import ArgumentError, ArgumentTypeError
import traceback


class PromptConstructionError(Exception):
    """Base class for all Loader related exceptions."""

    """Raised when prompt construction fails."""


class LoaderError(Exception):
    """Raised when loading data files fails."""


def catch(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a callable with universal error handling.

    Args:
        fn: Function or method to wrap.

    Returns:
        Wrapped callable that prints a friendly error message and exits.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Executes the wrapped loader function and catches generic exceptions.

        Args:
            *args: Positional arguments for the wrapped function.
            **kwargs: Keyword arguments for the wrapped function.

        Raises any (im)possible error.

        Returns:
            Any: The original return value of the wrapped function.
        """
        try:
            return fn(*args, **kwargs)
        except JSONDecodeError as e:
            print(f"JSON error: {e}")
            quit(1)
        except FileNotFoundError as e:
            print(f"Error: File missing: {e}")
            quit(1)
        except ImportError as e:
            print(f"Import error: {e}")
            quit(1)
        except PromptConstructionError as e:
            print(f"Error during prompt construction: {e}")
            quit(1)
        except LoaderError as e:
            print(f"Loader error: {e}")
            quit(1)
        except (ArgumentTypeError, ArgumentError) as e:
            print(f"Argument parser error: {e}")
            quit(1)
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
            traceback.print_exc()
            quit(1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt. Bye!")
            quit(0)
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            quit(1)

    return wrapper

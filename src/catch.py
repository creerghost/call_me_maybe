from typing import Callable, TypeVar, Any
from json import JSONDecodeError

T = TypeVar("T")


def catch(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    try:
        try:
            return fn(*args, **kwargs)
        except JSONDecodeError as e:
            raise Exception(f"JSON error: {e}")
        except FileNotFoundError as e:
            raise Exception(f"Error: File missing: {e}")
        except ImportError as e:
            raise Exception(f"Import error: {e}")
        except PromptConstructionError as e:
            raise Exception(f"Error during prompt construction: {e}")
        except KeyboardInterrupt:
            print("\nKeyboard interrupt. Bye!")
            quit(0)
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")
    except Exception as e:
        print(e)
        quit(1)


class PromptConstructionError(Exception):
    pass
